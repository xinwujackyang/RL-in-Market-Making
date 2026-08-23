from __future__ import annotations

import argparse
import csv
import sys
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


CONDITIONS = (
    ("current_ppo", 0.2),
    ("paper_aligned_ppo", 0.3),
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


def make_env_factory(cfg: Config, seed: int, stream_offset: int = 0):
    def env_factory(episode: int) -> TwoDealerMarketEnv:
        competitor = RandomMarketMaker(
            eps_bid_max=1.0,
            eps_ask_max=1.0,
            hedge_fraction_max=0.3,
            seed=seed * 100_000 + stream_offset + 20_000 + episode,
        )
        return TwoDealerMarketEnv(
            competitor,
            cfg,
            seed=seed * 100_000 + stream_offset + 10_000 + episode,
            reward_mode="total",
        )

    return env_factory


def train_seed(
    condition: str,
    clip_eps: float,
    seed: int,
    rollouts: int,
    horizon: int,
    eval_episodes: int,
    eval_days: int,
    trajectory_eval_episodes: int,
    trajectory_eval_days: int,
) -> tuple[dict, list[dict]]:
    cfg = Config(
        seed=seed,
        rollout=rollouts,
        horizon=horizon,
        eval_every_steps=horizon,
        eval_episodes=eval_episodes,
        eval_days=eval_days,
        num_investors=20,
        order_size_mode="unit",
        buy_probability=0.5,
        hidden_size=256,
        hidden_layers=2,
        minibatch_size=256,
        lr=5e-5,
        clip_eps=clip_eps,
    )
    training_factory = make_env_factory(cfg, seed)
    trajectory_factory = make_env_factory(cfg, seed, stream_offset=40_000)
    agent = PPOAgent(training_factory(0), cfg)
    trajectory_steps = cfg.steps_per_day * trajectory_eval_days

    def trajectory_evaluator(current: PPOAgent) -> dict:
        return evaluate_policy(
            current,
            trajectory_factory,
            trajectory_eval_episodes,
            trajectory_steps,
            deterministic=True,
        )[0]

    history = agent.train(evaluator=trajectory_evaluator)
    diagnostics_by_step = {int(row["step"]): row for row in history.diagnostics}
    trajectory_rows = []
    for evaluation in history.evaluation:
        step = int(evaluation["step"])
        diagnostic = diagnostics_by_step[step]
        trajectory_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "training_step": step,
                "deterministic_epsilon_bid_mean": evaluation["mean_epsilon_bid"],
                "deterministic_epsilon_ask_mean": evaluation["mean_epsilon_ask"],
                "deterministic_epsilon_bid_state_std": evaluation["epsilon_bid_std"],
                "deterministic_epsilon_ask_state_std": evaluation["epsilon_ask_std"],
                "latent_sigma_bid": evaluation["latent_std_bid"],
                "latent_sigma_ask": evaluation["latent_std_ask"],
                "entropy_proxy": diagnostic["entropy_proxy"],
                "approx_kl": diagnostic["approx_kl"],
                "clip_fraction": diagnostic["clip_fraction"],
                "policy_loss": diagnostic["policy_loss"],
                "value_loss": diagnostic["value_loss"],
            }
        )

    final_factory = make_env_factory(cfg, seed, stream_offset=80_000)
    final_steps = cfg.steps_per_day * cfg.eval_days
    deterministic_metrics, _ = evaluate_policy(
        agent, final_factory, cfg.eval_episodes, final_steps, deterministic=True
    )
    seed_everything(seed * 100_000 + 90_000)
    sampled_metrics, _ = evaluate_policy(
        agent, final_factory, cfg.eval_episodes, final_steps, deterministic=False
    )
    metrics = {
        "condition": condition,
        "seed": seed,
        "sampled_epsilon_bid_mean": sampled_metrics["mean_epsilon_bid"],
        "sampled_epsilon_ask_mean": sampled_metrics["mean_epsilon_ask"],
        "sampled_epsilon_bid_std": sampled_metrics["epsilon_bid_std"],
        "sampled_epsilon_ask_std": sampled_metrics["epsilon_ask_std"],
        "deterministic_epsilon_bid_mean": deterministic_metrics["mean_epsilon_bid"],
        "deterministic_epsilon_ask_mean": deterministic_metrics["mean_epsilon_ask"],
        "deterministic_epsilon_bid_state_std": deterministic_metrics["epsilon_bid_std"],
        "deterministic_epsilon_ask_state_std": deterministic_metrics["epsilon_ask_std"],
        "deterministic_error_bid": abs(deterministic_metrics["mean_epsilon_bid"]),
        "deterministic_error_ask": abs(deterministic_metrics["mean_epsilon_ask"]),
        "latent_sigma_bid": sampled_metrics["latent_std_bid"],
        "latent_sigma_ask": sampled_metrics["latent_std_ask"],
        "market_share": sampled_metrics["mean_market_share"],
        "spread_pnl_per_step": sampled_metrics["mean_spread_pnl"],
        "inventory_std": sampled_metrics["inventory_std"],
        "total_pnl": sampled_metrics["mean_total_pnl"],
        "total_pnl_per_step": sampled_metrics["mean_total_pnl_per_step"],
        "learning_rate": cfg.lr,
        "ppo_clip": cfg.clip_eps,
        "network_hidden_layers": cfg.hidden_layers,
        "network_hidden_size": cfg.hidden_size,
        "network_activation": "tanh",
        "minibatch_size": cfg.minibatch_size,
        "n_epochs": cfg.n_epochs,
        "rollout_length": cfg.horizon,
        "num_rollouts": cfg.rollout,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "num_investors": cfg.num_investors,
        "order_size_mode": cfg.order_size_mode,
        "buy_probability": cfg.buy_probability,
    }
    return metrics, trajectory_rows


def condition_rows(rows: list[dict], condition: str) -> list[dict]:
    return sorted(
        [row for row in rows if row["condition"] == condition],
        key=lambda row: (int(row["seed"]), int(row.get("training_step", 0))),
    )


def plot_mean_policy(trajectory: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for row_index, (condition, _) in enumerate(CONDITIONS):
        selected = condition_rows(trajectory, condition)
        seeds = sorted({int(row["seed"]) for row in selected})
        for column, side in enumerate(("bid", "ask")):
            axis = axes[row_index, column]
            for seed in seeds:
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                axis.plot(
                    [int(row["training_step"]) for row in seed_rows],
                    [float(row[f"deterministic_epsilon_{side}_mean"]) for row in seed_rows],
                    marker=".",
                    linewidth=1.2,
                    label=f"seed {seed}",
                )
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="epsilon*=0")
            axis.set(
                title=f"{condition}: deterministic {side}",
                xlabel="Training step",
                ylabel="Mean epsilon",
            )
            axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_policy_std(trajectory: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for row_index, (condition, _) in enumerate(CONDITIONS):
        selected = condition_rows(trajectory, condition)
        seeds = sorted({int(row["seed"]) for row in selected})
        for column, side in enumerate(("bid", "ask")):
            axis = axes[row_index, column]
            for seed in seeds:
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                axis.plot(
                    [int(row["training_step"]) for row in seed_rows],
                    [float(row[f"latent_sigma_{side}"]) for row in seed_rows],
                    marker=".",
                    linewidth=1.2,
                    label=f"seed {seed}",
                )
            axis.axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="initial sigma=1")
            axis.set(
                title=f"{condition}: latent sigma {side}",
                xlabel="Training step",
                ylabel="Latent sigma",
            )
            axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _bar_with_seed_points(axis, metrics: list[dict], keys: tuple[str, ...], title: str) -> None:
    conditions = [item[0] for item in CONDITIONS]
    x = np.arange(len(conditions), dtype=float)
    width = 0.68 / len(keys)
    for key_index, key in enumerate(keys):
        means = []
        stds = []
        groups = []
        for condition in conditions:
            values = np.asarray(
                [float(row[key]) for row in condition_rows(metrics, condition)], dtype=float
            )
            groups.append(values)
            means.append(values.mean())
            stds.append(values.std())
        offset = (key_index - (len(keys) - 1) / 2.0) * width
        positions = x + offset
        axis.bar(positions, means, width, yerr=stds, capsize=3, alpha=0.72, label=key)
        for position, values in zip(positions, groups):
            jitter = np.linspace(-0.22 * width, 0.22 * width, len(values))
            axis.scatter(position + jitter, values, color="black", s=13, zorder=3)
    axis.set_xticks(x, [name.replace("_", "\n") for name in conditions])
    axis.set_title(title)
    axis.legend(fontsize=7)


def plot_comparison(metrics: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    _bar_with_seed_points(
        axes[0, 0], metrics,
        ("deterministic_epsilon_bid_mean", "deterministic_epsilon_ask_mean"),
        "Deterministic pricing mean",
    )
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    _bar_with_seed_points(
        axes[0, 1], metrics,
        ("deterministic_error_bid", "deterministic_error_ask"),
        "Absolute deterministic error vs 0",
    )
    _bar_with_seed_points(
        axes[0, 2], metrics,
        ("sampled_epsilon_bid_std", "sampled_epsilon_ask_std"),
        "Sampled epsilon std",
    )
    _bar_with_seed_points(
        axes[1, 0], metrics,
        ("latent_sigma_bid", "latent_sigma_ask"),
        "Final latent sigma",
    )

    conditions = [item[0] for item in CONDITIONS]
    dispersion = []
    for condition in conditions:
        selected = condition_rows(metrics, condition)
        dispersion.append(
            (
                np.std([float(row["deterministic_epsilon_bid_mean"]) for row in selected]),
                np.std([float(row["deterministic_epsilon_ask_mean"]) for row in selected]),
            )
        )
    x = np.arange(len(conditions), dtype=float)
    width = 0.32
    axes[1, 1].bar(x - width / 2, [item[0] for item in dispersion], width, label="bid")
    axes[1, 1].bar(x + width / 2, [item[1] for item in dispersion], width, label="ask")
    axes[1, 1].set_xticks(x, [name.replace("_", "\n") for name in conditions])
    axes[1, 1].set_title("Across-seed deterministic mean dispersion")
    axes[1, 1].legend()

    _bar_with_seed_points(
        axes[1, 2], metrics,
        ("market_share", "spread_pnl_per_step", "total_pnl_per_step"),
        "Secondary market metrics",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit-flow current vs paper-aligned PPO")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--trajectory-eval-episodes", type=int, default=3)
    parser.add_argument("--trajectory-eval-days", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper_aligned_ppo")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_by_seed.csv"
    trajectory_path = args.output_dir / "training_trajectories.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    trajectory = read_csv(trajectory_path) if args.resume else []
    completed = {(row["condition"], int(row["seed"])) for row in metrics}

    for condition, clip_eps in CONDITIONS:
        for seed in range(args.seeds):
            if (condition, seed) in completed:
                print(f"Skipping completed {condition} seed={seed}", flush=True)
                continue
            print(f"Training {condition} seed={seed} (clip={clip_eps})", flush=True)
            row, trajectory_rows = train_seed(
                condition,
                clip_eps,
                seed,
                args.rollouts,
                args.horizon,
                args.eval_episodes,
                args.eval_days,
                args.trajectory_eval_episodes,
                args.trajectory_eval_days,
            )
            metrics.append(row)
            trajectory.extend(trajectory_rows)
            write_csv(metrics_path, metrics)
            write_csv(trajectory_path, trajectory)
            print(
                f"Finished {condition} seed={seed}: det eps="
                f"({row['deterministic_epsilon_bid_mean']:.3f}, "
                f"{row['deterministic_epsilon_ask_mean']:.3f}), latent sigma="
                f"({row['latent_sigma_bid']:.3f}, {row['latent_sigma_ask']:.3f})",
                flush=True,
            )

    plot_mean_policy(trajectory, args.output_dir / "mean_policy_training.png")
    plot_policy_std(trajectory, args.output_dir / "policy_std_training.png")
    plot_comparison(metrics, args.output_dir / "current_vs_paper_aligned.png")
    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
