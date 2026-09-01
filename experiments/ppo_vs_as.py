from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avellaneda_stoikov import AvellanedaStoikovMarketMaker
from config import Config
from environments import TwoDealerMarketEnv
from market import reference_spread
from ppo import PPOAgent


SEEDS = (0, 1, 2)
CHECKPOINTS = (20_480, 60_416, 100_352, 150_528, 204_800)
HORIZON = 1_024
ROLLOUTS = 200
EVAL_EPISODES = 10
EVAL_DAYS = 20
OUTPUT = ROOT / "results" / "ppo_vs_as"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_config(seed: int, rollouts: int = ROLLOUTS) -> Config:
    """The frozen reference PPO configuration, with only the opponent changed."""

    return Config(
        seed=seed,
        rollout=rollouts,
        horizon=HORIZON,
        relative_price=True,
        include_fill_feedback=False,
        paper_pnl_observation=False,
        observation_mode="default",
        num_investors=20,
        order_size_mode="unit",
        buy_probability=0.5,
        mu=0.0,
        sigma=0.2,
        hidden_size=256,
        hidden_layers=2,
        minibatch_size=256,
        n_epochs=10,
        lr=5e-5,
        clip_eps=0.3,
        gamma=0.999,
        gae_lambda=0.95,
        gae_value_target=False,
        ent_coef=0.003,
        vf_coef=0.5,
        vf_clip_param=None,
        kl_coef=0.0,
        max_grad_norm=0.5,
        state_dependent_std=True,
        policy_distribution="squashed_normal",
    )


def _metric_row(
    *,
    seed: int,
    checkpoint: int,
    agent: str,
    series: dict[str, list[float]],
    diagnostics: dict,
) -> dict:
    arrays = {key: np.asarray(value, dtype=float) for key, value in series.items()}
    if not np.allclose(
        arrays["spread"] + arrays["inventory_pnl"] - arrays["hedge_cost"],
        arrays["total"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(f"{agent} evaluation PnL identity failed")
    volume = float(arrays["volume"].sum())
    normalized_notional = float(arrays["normalized_notional"].sum())
    if volume <= 0.0 or normalized_notional <= 0.0:
        raise RuntimeError(f"{agent} captured no evaluation volume")
    epsilon_bid = arrays["epsilon_bid"]
    epsilon_ask = arrays["epsilon_ask"]
    return {
        "seed": seed,
        "training_step": checkpoint,
        "agent": agent,
        "evaluation_steps": len(arrays["total"]),
        "market_share": volume / (20.0 * len(arrays["total"])),
        "captured_volume_per_step": float(arrays["volume"].mean()),
        "spread_pnl_per_step": float(arrays["spread"].mean()),
        "spread_pnl_per_captured_unit": float(arrays["spread"].sum() / volume),
        "normalized_spread_monetization": float(
            arrays["spread"].sum() / normalized_notional
        ),
        "inventory_pnl_per_step": float(arrays["inventory_pnl"].mean()),
        "hedge_cost_per_step": float(arrays["hedge_cost"].mean()),
        "total_pnl_per_step": float(arrays["total"].mean()),
        "mean_inventory": float(arrays["inventory"].mean()),
        "mean_abs_inventory": float(np.abs(arrays["inventory"]).mean()),
        "inventory_std": float(arrays["inventory"].std()),
        "max_abs_inventory": float(np.abs(arrays["inventory"]).max()),
        "mean_epsilon_bid": float(epsilon_bid.mean()),
        "mean_epsilon_ask": float(epsilon_ask.mean()),
        "mean_symmetric_quote_level": float(((epsilon_bid + epsilon_ask) / 2).mean()),
        "mean_quote_skew": float((epsilon_bid - epsilon_ask).mean()),
        "mean_hedge_fraction": float(arrays["hedge_fraction"].mean()),
        "epsilon_near_bound_frequency": float(
            np.concatenate((np.abs(epsilon_bid) > 0.99, np.abs(epsilon_ask) > 0.99)).mean()
        ),
        "bid_clip_frequency": float(arrays["bid_clipped"].mean()),
        "ask_clip_frequency": float(arrays["ask_clipped"].mean()),
        "side_clip_frequency": float(
            np.concatenate((arrays["bid_clipped"], arrays["ask_clipped"])).mean()
        ),
        "approx_kl": float(diagnostics.get("approx_kl", np.nan)),
        "clip_fraction": float(diagnostics.get("clip_fraction", np.nan)),
        "value_loss": float(diagnostics.get("value_loss", np.nan)),
        "latent_std_bid": float(diagnostics.get("latent_std_bid", np.nan)),
        "latent_std_ask": float(diagnostics.get("latent_std_ask", np.nan)),
        "latent_std_hedge": float(diagnostics.get("latent_std_hedge", np.nan)),
    }


def evaluate_checkpoint(
    agent: PPOAgent,
    *,
    seed: int,
    checkpoint: int,
    episodes: int,
    days: int,
    diagnostics: dict,
) -> tuple[list[dict], list[dict]]:
    cfg = agent.cfg
    steps_per_episode = days * cfg.steps_per_day
    series: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("PPO", "A-S")
    }
    samples: list[dict] = []

    for episode in range(episodes):
        maker = AvellanedaStoikovMarketMaker(
            risk_horizon_steps=26,
            inventory_anchor=20.0,
            neutral_epsilon=0.0,
            hedge_fraction=0.0,
            cfg=cfg,
        )
        environment = TwoDealerMarketEnv(
            maker,
            cfg,
            seed=seed * 1_000_000 + 140_000 + episode,
            reward_mode="total",
        )
        observation = environment.reset()
        for episode_step in range(steps_per_episode):
            ppo_inventory_before = float(environment.inventory[0])
            action = agent.deterministic_action(observation)
            distance = reference_spread(environment.price, 1.0, cfg)
            observation, _, _, info = environment.step(action)
            as_quote = maker.last_quote
            if as_quote is None:
                raise RuntimeError("A-S evaluation action was not recorded")
            volumes = {
                "PPO": info["gross_investor_volume"] - info["competitor_gross_volume"],
                "A-S": info["competitor_gross_volume"],
            }
            actions = {
                "PPO": (float(action[0]), float(action[1]), float(action[2])),
                "A-S": (
                    info["competitor_epsilon_bid"],
                    info["competitor_epsilon_ask"],
                    info["competitor_hedge_fraction"],
                ),
            }
            for name, prefix in (("PPO", ""), ("A-S", "competitor_")):
                target = series[name]
                target["spread"].append(info[f"{prefix}spread_pnl"])
                target["inventory_pnl"].append(info[f"{prefix}inventory_pnl"])
                target["hedge_cost"].append(info[f"{prefix}hedge_cost"])
                target["total"].append(
                    info["total_pnl" if name == "PPO" else "competitor_pnl"]
                )
                target["inventory"].append(
                    info["inventory" if name == "PPO" else "competitor_inventory"]
                )
                target["volume"].append(volumes[name])
                target["normalized_notional"].append(volumes[name] * distance)
                target["epsilon_bid"].append(actions[name][0])
                target["epsilon_ask"].append(actions[name][1])
                target["hedge_fraction"].append(actions[name][2])
                target["bid_clipped"].append(
                    float(as_quote.bid_clipped) if name == "A-S" else 0.0
                )
                target["ask_clipped"].append(
                    float(as_quote.ask_clipped) if name == "A-S" else 0.0
                )
            if checkpoint == CHECKPOINTS[-1]:
                samples.append(
                    {
                        "seed": seed,
                        "training_step": checkpoint,
                        "episode": episode,
                        "episode_step": episode_step,
                        "inventory_before_action": ppo_inventory_before,
                        "epsilon_bid": float(action[0]),
                        "epsilon_ask": float(action[1]),
                        "quote_skew": float(action[0] - action[1]),
                        "symmetric_quote_level": float((action[0] + action[1]) / 2.0),
                        "hedge_fraction": float(action[2]),
                    }
                )

    rows = [
        _metric_row(
            seed=seed,
            checkpoint=checkpoint,
            agent=name,
            series=series[name],
            diagnostics=diagnostics if name == "PPO" else {},
        )
        for name in ("PPO", "A-S")
    ]
    return rows, samples


def run_seed(
    seed: int,
    *,
    rollouts: int,
    checkpoints: tuple[int, ...],
    eval_episodes: int,
    eval_days: int,
) -> tuple[list[dict], list[dict]]:
    cfg = make_config(seed, rollouts)
    maker = AvellanedaStoikovMarketMaker(cfg=cfg)
    environment = TwoDealerMarketEnv(
        maker,
        cfg,
        seed=seed * 100_000 + 14_000,
        reward_mode="total",
    )
    agent = PPOAgent(environment, cfg)
    evaluation_rows: list[dict] = []
    final_samples: list[dict] = []
    evaluation_index = 0

    def evaluator(current_agent: PPOAgent) -> dict:
        nonlocal evaluation_index
        checkpoint = checkpoints[evaluation_index]
        diagnostics = current_agent.history.diagnostics[-1]
        rows, samples = evaluate_checkpoint(
            current_agent,
            seed=seed,
            checkpoint=checkpoint,
            episodes=eval_episodes,
            days=eval_days,
            diagnostics=diagnostics,
        )
        evaluation_rows.extend(rows)
        final_samples.extend(samples)
        evaluation_index += 1
        return {"mean_total_pnl": rows[0]["total_pnl_per_step"]}

    agent.train(evaluator=evaluator, evaluation_steps=set(checkpoints))
    if evaluation_index != len(checkpoints):
        raise RuntimeError("not every requested PPO checkpoint was evaluated")
    return evaluation_rows, final_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reference PPO from scratch against A-S")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-days", type=int, default=EVAL_DAYS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.rollouts, args.eval_episodes, args.eval_days) <= 0:
        raise ValueError("rollouts, eval episodes and eval days must be positive")

    total_steps = args.rollouts * HORIZON
    checkpoints = CHECKPOINTS if total_steps == CHECKPOINTS[-1] else (total_steps,)
    if any(checkpoint % HORIZON for checkpoint in checkpoints):
        raise RuntimeError("checkpoints must align with PPO rollout boundaries")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics_by_seed_checkpoint.csv"
    samples_path = args.output / "final_policy_samples.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    samples = read_csv(samples_path) if args.resume else []
    completed = {
        seed
        for seed in args.seeds
        if sum(int(row["seed"]) == seed for row in metrics) == 2 * len(checkpoints)
    }

    for seed in args.seeds:
        if seed in completed:
            print(f"Skipping completed seed={seed}", flush=True)
            continue
        metrics = [row for row in metrics if int(row["seed"]) != seed]
        samples = [row for row in samples if int(row["seed"]) != seed]
        print(f"Training PPO vs A-S seed={seed} for {total_steps:,} steps", flush=True)
        seed_metrics, seed_samples = run_seed(
            seed,
            rollouts=args.rollouts,
            checkpoints=checkpoints,
            eval_episodes=args.eval_episodes,
            eval_days=args.eval_days,
        )
        metrics.extend(seed_metrics)
        samples.extend(seed_samples)
        write_csv(metrics_path, metrics)
        if samples:
            write_csv(samples_path, samples)
        final_ppo = seed_metrics[-2]
        final_as = seed_metrics[-1]
        print(
            f"Finished seed={seed}: PPO total={final_ppo['total_pnl_per_step']:.4f} "
            f"A-S total={final_as['total_pnl_per_step']:.4f}",
            flush=True,
        )
    print(f"Saved PPO-vs-A-S data under {args.output}")


if __name__ == "__main__":
    main()
