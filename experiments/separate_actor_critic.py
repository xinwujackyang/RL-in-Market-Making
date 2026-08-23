from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import read_csv, train_seed, write_csv


def plot_shared_vs_separate(
    shared_trajectory: list[dict],
    separate_trajectory: list[dict],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    groups = (
        ("Shared backbone", shared_trajectory),
        ("Separate actor/critic", separate_trajectory),
    )
    for row_index, (label, rows) in enumerate(groups):
        seeds = sorted({int(row["seed"]) for row in rows})
        for column, side in enumerate(("bid", "ask")):
            axis = axes[row_index, column]
            for seed in seeds:
                selected = sorted(
                    [row for row in rows if int(row["seed"]) == seed],
                    key=lambda row: int(row["training_step"]),
                )
                axis.plot(
                    [int(row["training_step"]) for row in selected],
                    [float(row[f"deterministic_epsilon_{side}_mean"]) for row in selected],
                    marker="o",
                    linewidth=1.3,
                    label=f"seed {seed}",
                )
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="epsilon*=0")
            axis.set(
                title=f"{label}: deterministic {side}",
                xlabel="Training step",
                ylabel="Mean epsilon",
            )
            axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-run PPO with separate actor and critic MLPs")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--trajectory-eval-episodes", type=int, default=3)
    parser.add_argument("--trajectory-eval-days", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "separate_actor_critic",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_by_seed.csv"
    trajectory_path = args.output_dir / "training_trajectories.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    trajectory = read_csv(trajectory_path) if args.resume else []
    completed = {int(row["seed"]) for row in metrics}

    for seed in range(args.seeds):
        if seed in completed:
            print(f"Skipping completed seed={seed}", flush=True)
            continue
        print(f"Training separate actor/critic seed={seed}", flush=True)
        row, trajectory_rows = train_seed(
            seed,
            args.rollouts,
            args.horizon,
            args.eval_episodes,
            args.eval_days,
            args.trajectory_eval_episodes,
            args.trajectory_eval_days,
        )
        row["network_architecture"] = "separate_actor_critic"
        for trajectory_row in trajectory_rows:
            trajectory_row["network_architecture"] = "separate_actor_critic"
        metrics.append(row)
        trajectory.extend(trajectory_rows)
        write_csv(metrics_path, metrics)
        write_csv(trajectory_path, trajectory)
        print(
            f"Finished seed={seed}: det eps="
            f"({row['deterministic_epsilon_bid_mean']:.3f}, "
            f"{row['deterministic_epsilon_ask_mean']:.3f}), latent sigma="
            f"({row['latent_sigma_bid']:.3f}, {row['latent_sigma_ask']:.3f})",
            flush=True,
        )

    shared_path = ROOT / "results" / "long_run_random" / "training_trajectories.csv"
    plot_shared_vs_separate(
        read_csv(shared_path),
        trajectory,
        args.output_dir / "mean_policy_shared_vs_separate.png",
    )
    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
