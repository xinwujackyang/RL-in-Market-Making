from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import make_env_factory, read_csv, write_csv

sys.path.insert(0, str(ROOT / "src"))

from config import Config, seed_everything
from evaluation import evaluate_policy
from ppo import PPOAgent


CONFIRMATION_CHECKPOINTS = {
    20_480,
    60_416,
    100_352,
    149_504,
    199_680,
    204_800,
}


def train_seed(
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
        n_epochs=3,
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
        eval_episodes=eval_episodes,
        eval_days=eval_days,
    )
    if cfg.total_steps != 204_800:
        raise ValueError("epochs=3 confirmation must run exactly 204,800 steps")
    if max(CONFIRMATION_CHECKPOINTS) != cfg.total_steps:
        raise ValueError("final confirmation checkpoint must equal total training steps")

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

    history = agent.train(
        evaluator=trajectory_evaluator,
        evaluation_steps=CONFIRMATION_CHECKPOINTS,
    )
    diagnostics_by_step = {int(row["step"]): row for row in history.diagnostics}
    trajectory_rows = []
    for evaluation in history.evaluation:
        step = int(evaluation["step"])
        diagnostic = diagnostics_by_step[step]
        bid = evaluation["mean_epsilon_bid"]
        ask = evaluation["mean_epsilon_ask"]
        trajectory_rows.append(
            {
                "seed": seed,
                "training_step": step,
                "deterministic_epsilon_bid_mean": bid,
                "deterministic_epsilon_ask_mean": ask,
                "two_side_mae": (abs(bid) + abs(ask)) / 2.0,
                "approx_kl": diagnostic["approx_kl"],
                "clip_fraction": diagnostic["clip_fraction"],
                "policy_loss": diagnostic["policy_loss"],
                "value_loss": diagnostic["value_loss"],
                "latent_sigma_bid_mean": evaluation["latent_std_bid"],
                "latent_sigma_ask_mean": evaluation["latent_std_ask"],
                "latent_sigma_bid_state_std": evaluation[
                    "latent_std_bid_state_std"
                ],
                "latent_sigma_ask_state_std": evaluation[
                    "latent_std_ask_state_std"
                ],
                "n_epochs": cfg.n_epochs,
                "ent_coef": cfg.ent_coef,
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
    deterministic_bid = deterministic_metrics["mean_epsilon_bid"]
    deterministic_ask = deterministic_metrics["mean_epsilon_ask"]
    metrics = {
        "seed": seed,
        "training_steps": cfg.total_steps,
        "deterministic_epsilon_bid_mean": deterministic_bid,
        "deterministic_epsilon_ask_mean": deterministic_ask,
        "two_side_mae": (abs(deterministic_bid) + abs(deterministic_ask)) / 2.0,
        "sampled_epsilon_bid_std": sampled_metrics["epsilon_bid_std"],
        "sampled_epsilon_ask_std": sampled_metrics["epsilon_ask_std"],
        "latent_sigma_bid_mean": sampled_metrics["latent_std_bid"],
        "latent_sigma_ask_mean": sampled_metrics["latent_std_ask"],
        "latent_sigma_bid_state_std": sampled_metrics[
            "latent_std_bid_state_std"
        ],
        "latent_sigma_ask_state_std": sampled_metrics[
            "latent_std_ask_state_std"
        ],
        "market_share": sampled_metrics["mean_market_share"],
        "spread_pnl_per_step": sampled_metrics["mean_spread_pnl"],
        "total_pnl_per_step": sampled_metrics["mean_total_pnl_per_step"],
        "inventory_std": sampled_metrics["inventory_std"],
        "n_epochs": cfg.n_epochs,
        "ent_coef": cfg.ent_coef,
    }
    return metrics, trajectory_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-seed epochs=3 confirmation")
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
        default=ROOT / "results" / "epochs_3_confirmation",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.seeds != 5:
        raise ValueError("epochs=3 confirmation requires exactly five seeds")

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
        print(f"Training epochs=3 confirmation seed={seed}", flush=True)
        row, trajectory_rows = train_seed(
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
            f"Finished seed={seed}: det eps="
            f"({row['deterministic_epsilon_bid_mean']:.3f}, "
            f"{row['deterministic_epsilon_ask_mean']:.3f}), MAE="
            f"{row['two_side_mae']:.3f}, latent sigma="
            f"({row['latent_sigma_bid_mean']:.3f}, "
            f"{row['latent_sigma_ask_mean']:.3f})",
            flush=True,
        )

    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
