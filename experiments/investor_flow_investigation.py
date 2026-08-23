from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import RandomMarketMaker
from config import Config, seed_everything
from environments import TwoDealerMarketEnv
from evaluation import evaluate_policy
from ppo import PPOAgent


CURRENT_PPO = {
    "sigma": 0.2,
    "hidden_size": 256,
    "hidden_layers": 3,
    "clip_eps": 0.2,
    "minibatch_size": 256,
    "lr": 5e-5,
}

PAPER_PPO = {
    "sigma": 0.1,
    "hidden_size": 256,
    "hidden_layers": 2,
    "clip_eps": 0.3,
    "minibatch_size": 256,
    "lr": 5e-5,
}

REPORTED_METRICS = (
    "mean_epsilon_bid",
    "mean_epsilon_ask",
    "epsilon_bid_std",
    "epsilon_ask_std",
    "latent_std_bid",
    "latent_std_ask",
    "analytical_error_bid",
    "analytical_error_ask",
    "deterministic_mean_epsilon_bid",
    "deterministic_mean_epsilon_ask",
    "mean_market_share",
    "mean_spread_pnl",
    "spread_pnl_std",
    "inventory_std",
    "mean_total_pnl",
    "mean_total_pnl_per_step",
    "gross_investor_volume_mean",
    "gross_investor_volume_std",
    "net_investor_flow_mean",
    "net_investor_flow_std",
    "n_buy_mean",
    "n_buy_std",
    "n_sell_mean",
    "n_sell_std",
    "n_rl_won_mean",
    "n_rl_won_std",
    "n_competitor_won_mean",
    "n_competitor_won_std",
    "market_share_std",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def conditions_for_phase(phase: str) -> list[dict]:
    flow_only = [
        {"condition": "gamma_current_ppo", "order_size_mode": "gamma", "ppo": CURRENT_PPO},
        {"condition": "unit_current_ppo", "order_size_mode": "unit", "ppo": CURRENT_PPO},
    ]
    paper = [
        {"condition": "unit_paper_ppo", "order_size_mode": "unit", "ppo": PAPER_PPO},
    ]
    if phase == "flow-only":
        return flow_only
    if phase == "paper-ppo":
        return paper
    return flow_only + paper


def make_env_factory(cfg: Config, seed: int):
    def env_factory(episode: int) -> TwoDealerMarketEnv:
        competitor = RandomMarketMaker(
            eps_bid_max=1.0,
            eps_ask_max=1.0,
            hedge_fraction_max=0.3,
            seed=seed * 100_000 + 20_000 + episode,
        )
        return TwoDealerMarketEnv(
            competitor,
            cfg,
            seed=seed * 100_000 + 10_000 + episode,
            reward_mode="total",
        )

    return env_factory


def train_condition(
    condition: dict,
    seed: int,
    rollouts: int,
    horizon: int,
    eval_episodes: int,
    eval_days: int,
) -> tuple[dict, list[dict]]:
    ppo = condition["ppo"]
    cfg = Config(
        seed=seed,
        rollout=rollouts,
        horizon=horizon,
        eval_episodes=eval_episodes,
        eval_days=eval_days,
        num_investors=20,
        order_size_mode=condition["order_size_mode"],
        buy_probability=0.5,
        sigma=ppo["sigma"],
        hidden_size=ppo["hidden_size"],
        hidden_layers=ppo["hidden_layers"],
        clip_eps=ppo["clip_eps"],
        minibatch_size=ppo["minibatch_size"],
        lr=ppo["lr"],
    )
    env_factory = make_env_factory(cfg, seed)
    agent = PPOAgent(env_factory(0), cfg)
    history = agent.train()

    steps = cfg.steps_per_day * cfg.eval_days
    deterministic_metrics, _ = evaluate_policy(
        agent, env_factory, cfg.eval_episodes, steps, deterministic=True
    )
    # A fixed evaluation RNG makes action-sampling noise paired across conditions.
    seed_everything(seed * 100_000 + 30_000)
    metrics, _ = evaluate_policy(
        agent, env_factory, cfg.eval_episodes, steps, deterministic=False
    )
    metrics.update(
        condition=condition["condition"],
        seed=seed,
        order_size_mode=condition["order_size_mode"],
        num_investors=cfg.num_investors,
        buy_probability=cfg.buy_probability,
        ppo_sigma=cfg.sigma,
        ppo_hidden_size=cfg.hidden_size,
        ppo_hidden_layers=cfg.hidden_layers,
        ppo_clip_eps=cfg.clip_eps,
        ppo_minibatch_size=cfg.minibatch_size,
        ppo_lr=cfg.lr,
        ppo_rollouts=cfg.rollout,
        ppo_horizon=cfg.horizon,
        deterministic_mean_epsilon_bid=deterministic_metrics["mean_epsilon_bid"],
        deterministic_mean_epsilon_ask=deterministic_metrics["mean_epsilon_ask"],
        deterministic_epsilon_bid_state_std=deterministic_metrics["epsilon_bid_std"],
        deterministic_epsilon_ask_state_std=deterministic_metrics["epsilon_ask_std"],
        analytical_error_bid=abs(metrics["mean_epsilon_bid"]),
        analytical_error_ask=abs(metrics["mean_epsilon_ask"]),
        deterministic_analytical_error_bid=abs(deterministic_metrics["mean_epsilon_bid"]),
        deterministic_analytical_error_ask=abs(deterministic_metrics["mean_epsilon_ask"]),
    )

    trajectory = []
    for diagnostic in history.diagnostics:
        trajectory.append(
            {
                "condition": condition["condition"],
                "seed": seed,
                **diagnostic,
            }
        )
    return metrics, trajectory


def summarize(metrics: list[dict]) -> list[dict]:
    rows = []
    conditions = sorted({row["condition"] for row in metrics})
    for condition in conditions:
        selected = [row for row in metrics if row["condition"] == condition]
        for metric in REPORTED_METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "across_seed_mean": float(values.mean()),
                    "across_seed_std": float(values.std()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "n_seeds": len(values),
                }
            )
    return rows


def paired_differences(metrics: list[dict]) -> list[dict]:
    by_key = {(row["condition"], int(row["seed"])): row for row in metrics}
    comparisons = (
        ("gamma_current_ppo", "unit_current_ppo"),
        ("unit_current_ppo", "unit_paper_ppo"),
    )
    rows = []
    for baseline, treatment in comparisons:
        seeds = sorted(
            seed
            for condition, seed in by_key
            if condition == baseline and (treatment, seed) in by_key
        )
        for seed in seeds:
            for metric in REPORTED_METRICS:
                before = float(by_key[(baseline, seed)][metric])
                after = float(by_key[(treatment, seed)][metric])
                rows.append(
                    {
                        "baseline": baseline,
                        "treatment": treatment,
                        "seed": seed,
                        "metric": metric,
                        "baseline_value": before,
                        "treatment_value": after,
                        "paired_difference": after - before,
                    }
                )
    return rows


def plot_latent_std(trajectory: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in trajectory:
        grouped[(row["condition"], int(row["step"]))].append(row)
    for axis, side in zip(axes, ("bid", "ask")):
        for condition in sorted({row["condition"] for row in trajectory}):
            steps = sorted(step for name, step in grouped if name == condition)
            means = []
            stds = []
            for step in steps:
                values = np.asarray(
                    [float(row[f"latent_std_{side}"]) for row in grouped[(condition, step)]]
                )
                means.append(values.mean())
                stds.append(values.std())
            means_array = np.asarray(means)
            stds_array = np.asarray(stds)
            axis.plot(steps, means_array, label=condition)
            axis.fill_between(steps, means_array - stds_array, means_array + stds_array, alpha=0.16)
        axis.set(title=f"Latent std: {side}", xlabel="Training steps", ylabel="Latent std")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_summary(metrics: list[dict], output: Path) -> None:
    conditions = sorted({row["condition"] for row in metrics})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    specs = (
        (("mean_epsilon_bid", "mean_epsilon_ask"), "Sampled pricing mean"),
        (("epsilon_bid_std", "epsilon_ask_std"), "Sampled action std"),
        (("latent_std_bid", "latent_std_ask"), "Final latent std"),
        (("analytical_error_bid", "analytical_error_ask"), "Absolute error vs 0"),
        (("mean_market_share",), "RL market share"),
        (("mean_spread_pnl", "mean_total_pnl_per_step"), "PnL per step"),
    )
    x = np.arange(len(conditions))
    for axis, (keys, title) in zip(axes.flat, specs):
        width = 0.7 / len(keys)
        for key_index, key in enumerate(keys):
            means = []
            errors = []
            for condition in conditions:
                values = np.asarray(
                    [float(row[key]) for row in metrics if row["condition"] == condition]
                )
                means.append(values.mean())
                errors.append(values.std())
            offset = (key_index - (len(keys) - 1) / 2) * width
            axis.bar(x + offset, means, width, yerr=errors, capsize=3, label=key)
        axis.set_xticks(x, [name.replace("_", "\n") for name in conditions])
        axis.set_title(title)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned investor-flow A/B investigation")
    parser.add_argument("--phase", choices=("flow-only", "paper-ppo", "all"), default="flow-only")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = ROOT / "results" / "investor_flow"
    output.mkdir(parents=True, exist_ok=True)
    slug = args.phase.replace("-", "_")
    metrics_path = output / f"metrics_{slug}.csv"
    trajectory_path = output / f"training_{slug}.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    trajectory = read_csv(trajectory_path) if args.resume else []
    completed = {(row["condition"], int(row["seed"])) for row in metrics}

    conditions = conditions_for_phase(args.phase)
    for condition in conditions:
        for seed in range(args.seeds):
            key = (condition["condition"], seed)
            if key in completed:
                print(f"Skipping completed {condition['condition']} seed={seed}", flush=True)
                continue
            print(f"Training {condition['condition']} seed={seed}", flush=True)
            row, history_rows = train_condition(
                condition,
                seed,
                args.rollouts,
                args.horizon,
                args.eval_episodes,
                args.eval_days,
            )
            metrics.append(row)
            trajectory.extend(history_rows)
            write_csv(metrics_path, metrics)
            write_csv(trajectory_path, trajectory)
            print(
                f"Finished {condition['condition']} seed={seed}: "
                f"eps=({row['mean_epsilon_bid']:.3f}, {row['mean_epsilon_ask']:.3f}), "
                f"action_std=({row['epsilon_bid_std']:.3f}, {row['epsilon_ask_std']:.3f}), "
                f"latent_std=({row['latent_std_bid']:.3f}, {row['latent_std_ask']:.3f})",
                flush=True,
            )

    write_csv(output / f"summary_{slug}.csv", summarize(metrics))
    write_csv(output / f"paired_differences_{slug}.csv", paired_differences(metrics))
    plot_latent_std(trajectory, output / f"latent_std_trajectory_{slug}.png")
    plot_summary(metrics, output / f"comparison_{slug}.png")
    print(f"Saved phase outputs under {output}")


if __name__ == "__main__":
    main()
