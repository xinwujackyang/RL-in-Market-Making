from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import write_csv
from persistent_state_dependent_std import make_env_factory

sys.path.insert(0, str(ROOT / "src"))

from config import Config, seed_everything
from evaluation import evaluate_policy
from ppo import PPOAgent


def train_and_evaluate_seed(
    seed: int,
    rollouts: int,
    horizon: int,
    eval_episodes: int,
    eval_days: int,
) -> dict[str, np.ndarray]:
    cfg = Config(
        seed=seed,
        rollout=rollouts,
        horizon=horizon,
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
        eval_episodes=eval_episodes,
        eval_days=eval_days,
    )
    agent = PPOAgent(make_env_factory(cfg, seed)(0), cfg)
    agent.train()

    final_factory = make_env_factory(cfg, seed, stream_offset=80_000)
    seed_everything(seed * 100_000 + 90_000)
    _, data = evaluate_policy(
        agent,
        final_factory,
        cfg.eval_episodes,
        cfg.steps_per_day * cfg.eval_days,
        deterministic=False,
    )
    required = ("inventory_before_action", "latent_std_bid", "latent_std_ask")
    if any(key not in data for key in required):
        raise KeyError("evaluation data is missing inventory or latent std arrays")
    arrays = {key: data[key] for key in required}
    if len({len(values) for values in arrays.values()}) != 1:
        raise ValueError("evaluation arrays must have matching lengths")
    return arrays


def aggregate_bins(evaluations: list[dict[str, np.ndarray]]) -> list[dict]:
    pooled_inventory = np.concatenate(
        [evaluation["inventory_before_action"] for evaluation in evaluations]
    )
    inner_edges = np.quantile(pooled_inventory, [0.2, 0.4, 0.6, 0.8])
    rows = []
    for seed, evaluation in enumerate(evaluations):
        inventory = evaluation["inventory_before_action"]
        sigma_bid = evaluation["latent_std_bid"]
        sigma_ask = evaluation["latent_std_ask"]
        bin_indices = np.searchsorted(inner_edges, inventory, side="right")
        if len(bin_indices) != len(inventory) or np.any((bin_indices < 0) | (bin_indices > 4)):
            raise ValueError("each observation must be assigned to exactly one inventory bin")
        if np.bincount(bin_indices, minlength=5).sum() != len(inventory):
            raise ValueError("inventory bin counts do not cover every observation")
        for bin_index in range(5):
            selected = bin_indices == bin_index
            count = int(selected.sum())
            if count == 0:
                raise ValueError(f"seed {seed} has an empty inventory quantile bin")
            bid_mean = float(sigma_bid[selected].mean())
            ask_mean = float(sigma_ask[selected].mean())
            rows.append(
                {
                    "seed": seed,
                    "inventory_bin": f"Q{bin_index + 1}",
                    "inventory_mean": float(inventory[selected].mean()),
                    "sigma_bid_mean": bid_mean,
                    "sigma_ask_mean": ask_mean,
                    "sigma_bid_minus_ask": bid_mean - ask_mean,
                    "count": count,
                }
            )
    numeric_keys = (
        "inventory_mean",
        "sigma_bid_mean",
        "sigma_ask_mean",
        "sigma_bid_minus_ask",
    )
    if any(not np.isfinite(row[key]) for row in rows for key in numeric_keys):
        raise ValueError("inventory sigma bin output contains non-finite values")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze state-dependent policy variance across inventory quantiles"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "inventory_variance_analysis",
    )
    args = parser.parse_args()

    evaluations = []
    for seed in range(args.seeds):
        print(f"Training Persistent state-dependent seed={seed}", flush=True)
        evaluation = train_and_evaluate_seed(
            seed,
            args.rollouts,
            args.horizon,
            args.eval_episodes,
            args.eval_days,
        )
        evaluations.append(evaluation)
        print(
            f"Finished seed={seed}: {len(evaluation['inventory_before_action'])} "
            "stochastic evaluation observations",
            flush=True,
        )

    rows = aggregate_bins(evaluations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "inventory_sigma_bins.csv", rows)
    print(f"Saved {len(rows)} bin rows under {args.output_dir}")


if __name__ == "__main__":
    main()
