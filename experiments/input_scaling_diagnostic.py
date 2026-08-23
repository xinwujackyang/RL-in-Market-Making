from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import make_env_factory

sys.path.insert(0, str(ROOT / "src"))

from config import Config
from ppo import PPOAgent


FEATURES = ("inventory", "price", "total_pnl", "inventory_pnl", "hedge_cost")
CSV_FIELDS = (
    "type",
    "step",
    "feature",
    "mean",
    "std",
    "p01",
    "p50",
    "p99",
    "mean_abs_pre_activation",
    "tanh_saturation_fraction",
)


def write_stats(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def collect_diagnostics(
    agent: PPOAgent,
    env_factory,
    episodes: int,
    steps: int,
    training_step: int,
) -> tuple[list[dict], float]:
    observations = []
    episode_totals = []
    for episode in range(episodes):
        env = env_factory(episode)
        observation = env.reset()
        total = 0.0
        for _ in range(steps):
            observations.append(observation.copy())
            action = agent.deterministic_action(observation)
            observation, _, _, info = env.step(action)
            total += info["total_pnl"]
        episode_totals.append(total)

    observation_array = np.asarray(observations, dtype=np.float32)
    rows = []
    for index, feature in enumerate(FEATURES):
        values = observation_array[:, index].astype(np.float64)
        rows.append(
            {
                "type": "observation",
                "step": training_step,
                "feature": feature,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "p01": float(np.quantile(values, 0.01)),
                "p50": float(np.quantile(values, 0.50)),
                "p99": float(np.quantile(values, 0.99)),
            }
        )

    first_layer = agent.network.actor[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError("actor[0] must be the first Linear layer")
    with torch.no_grad():
        tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=agent.cfg.device,
        )
        pre_activation = first_layer(tensor)
        mean_abs = pre_activation.abs().mean().item()
        saturation = (torch.tanh(pre_activation).abs() > 0.95).float().mean().item()
    rows.append(
        {
            "type": "actor_activation",
            "step": training_step,
            "mean_abs_pre_activation": mean_abs,
            "tanh_saturation_fraction": saturation,
        }
    )
    return rows, float(np.mean(episode_totals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Observation scale and actor saturation diagnostic")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=59)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--checkpoints", type=int, nargs=2, default=(20_480, 60_416))
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-days", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "input_scaling_diagnostic",
    )
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        rollout=args.rollouts,
        horizon=args.horizon,
        num_investors=20,
        order_size_mode="unit",
        buy_probability=0.5,
        sigma=0.2,
        hidden_size=256,
        hidden_layers=2,
        minibatch_size=256,
        n_epochs=10,
        lr=5e-5,
        clip_eps=0.3,
    )
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if len(checkpoints) != 2:
        raise ValueError("exactly two distinct checkpoints are required")
    if any(step % cfg.horizon != 0 for step in checkpoints):
        raise ValueError("checkpoints must lie on rollout boundaries")
    if checkpoints[-1] != cfg.total_steps:
        raise ValueError("the final checkpoint must equal total training steps")

    training_factory = make_env_factory(cfg, args.seed)
    evaluation_factory = make_env_factory(cfg, args.seed, stream_offset=40_000)
    agent = PPOAgent(training_factory(0), cfg)
    evaluation_steps = cfg.steps_per_day * args.eval_days
    pending_steps = iter(checkpoints)
    stats_rows: list[dict] = []

    def evaluator(current: PPOAgent) -> dict:
        training_step = next(pending_steps)
        rows, mean_total_pnl = collect_diagnostics(
            current,
            evaluation_factory,
            args.eval_episodes,
            evaluation_steps,
            training_step,
        )
        stats_rows.extend(rows)
        return {"mean_total_pnl": mean_total_pnl}

    agent.train(evaluator=evaluator, evaluation_steps=set(checkpoints))
    if len(stats_rows) != 2 * (len(FEATURES) + 1):
        raise RuntimeError("expected five observation rows plus one activation row per checkpoint")
    numeric_values = [
        float(value)
        for row in stats_rows
        for key, value in row.items()
        if key not in {"type", "feature", "step"} and value != ""
    ]
    if not np.isfinite(numeric_values).all():
        raise RuntimeError("diagnostic statistics contain NaN or infinity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_stats(args.output_dir / "stats.csv", stats_rows)
    print(f"Saved {args.output_dir / 'stats.csv'}")


if __name__ == "__main__":
    main()
