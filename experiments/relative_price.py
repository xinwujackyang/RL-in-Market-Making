from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import make_env_factory, read_csv, rollout_checkpoints, write_csv

sys.path.insert(0, str(ROOT / "src"))

from config import Config, seed_everything
from evaluation import evaluate_policy
from ppo import PPOAgent


SATURATION_CHECKPOINTS = {20_480, 60_416}


def actor_first_layer_saturation(
    agent: PPOAgent,
    env_factory,
    episodes: int,
    steps: int,
) -> float:
    observations = []
    for episode in range(episodes):
        env = env_factory(episode)
        observation = env.reset()
        for _ in range(steps):
            observations.append(observation.copy())
            action = agent.deterministic_action(observation)
            observation, _, _, _ = env.step(action)

    first_layer = agent.network.actor[0]
    if not isinstance(first_layer, nn.Linear):
        raise TypeError("actor[0] must be the first Linear layer")
    with torch.no_grad():
        tensor = torch.as_tensor(
            np.asarray(observations, dtype=np.float32),
            dtype=torch.float32,
            device=agent.cfg.device,
        )
        activations = torch.tanh(first_layer(tensor))
    return float((activations.abs() > 0.95).float().mean().item())


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
        eval_episodes=eval_episodes,
        eval_days=eval_days,
    )
    checkpoints = rollout_checkpoints(cfg.horizon, cfg.total_steps)
    missing_saturation_steps = SATURATION_CHECKPOINTS - checkpoints
    if cfg.total_steps >= max(SATURATION_CHECKPOINTS) and missing_saturation_steps:
        raise ValueError(
            "saturation checkpoints must be training checkpoints: "
            f"{sorted(missing_saturation_steps)}"
        )

    training_factory = make_env_factory(cfg, seed)
    trajectory_factory = make_env_factory(cfg, seed, stream_offset=40_000)
    agent = PPOAgent(training_factory(0), cfg)
    trajectory_steps = cfg.steps_per_day * trajectory_eval_days
    evaluation_step_iterator = iter(sorted(checkpoints))
    saturation_by_step: dict[int, float] = {}

    def trajectory_evaluator(current: PPOAgent) -> dict:
        training_step = next(evaluation_step_iterator)
        metrics, _ = evaluate_policy(
            current,
            trajectory_factory,
            trajectory_eval_episodes,
            trajectory_steps,
            deterministic=True,
        )
        if training_step in SATURATION_CHECKPOINTS:
            saturation = actor_first_layer_saturation(
                current,
                trajectory_factory,
                trajectory_eval_episodes,
                trajectory_steps,
            )
            saturation_by_step[training_step] = saturation
            metrics["actor_first_layer_saturation"] = saturation
        return metrics

    history = agent.train(
        evaluator=trajectory_evaluator,
        evaluation_steps=checkpoints,
    )
    diagnostics_by_step = {int(row["step"]): row for row in history.diagnostics}
    trajectory_rows = []
    for evaluation in history.evaluation:
        step = int(evaluation["step"])
        diagnostic = diagnostics_by_step[step]
        trajectory_rows.append(
            {
                "seed": seed,
                "training_step": step,
                "deterministic_epsilon_bid_mean": evaluation["mean_epsilon_bid"],
                "deterministic_epsilon_ask_mean": evaluation["mean_epsilon_ask"],
                "deterministic_epsilon_bid_state_std": evaluation["epsilon_bid_std"],
                "deterministic_epsilon_ask_state_std": evaluation["epsilon_ask_std"],
                "latent_sigma_bid": evaluation["latent_std_bid"],
                "latent_sigma_ask": evaluation["latent_std_ask"],
                "actor_first_layer_saturation": evaluation.get(
                    "actor_first_layer_saturation", ""
                ),
                "entropy_proxy": diagnostic["entropy_proxy"],
                "approx_kl": diagnostic["approx_kl"],
                "clip_fraction": diagnostic["clip_fraction"],
                "policy_loss": diagnostic["policy_loss"],
                "value_loss": diagnostic["value_loss"],
                "relative_price": True,
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
        "seed": seed,
        "training_steps": cfg.total_steps,
        "relative_price": cfg.relative_price,
        "actor_first_layer_saturation_20480": saturation_by_step.get(20_480, ""),
        "actor_first_layer_saturation_60416": saturation_by_step.get(60_416, ""),
        "deterministic_epsilon_bid_mean": deterministic_metrics["mean_epsilon_bid"],
        "deterministic_epsilon_ask_mean": deterministic_metrics["mean_epsilon_ask"],
        "deterministic_error_bid": abs(deterministic_metrics["mean_epsilon_bid"]),
        "deterministic_error_ask": abs(deterministic_metrics["mean_epsilon_ask"]),
        "sampled_epsilon_bid_mean": sampled_metrics["mean_epsilon_bid"],
        "sampled_epsilon_ask_mean": sampled_metrics["mean_epsilon_ask"],
        "sampled_epsilon_bid_std": sampled_metrics["epsilon_bid_std"],
        "sampled_epsilon_ask_std": sampled_metrics["epsilon_ask_std"],
        "latent_sigma_bid": sampled_metrics["latent_std_bid"],
        "latent_sigma_ask": sampled_metrics["latent_std_ask"],
        "market_share": sampled_metrics["mean_market_share"],
        "spread_pnl_per_step": sampled_metrics["mean_spread_pnl"],
        "inventory_std": sampled_metrics["inventory_std"],
        "total_pnl": sampled_metrics["mean_total_pnl"],
        "total_pnl_per_step": sampled_metrics["mean_total_pnl_per_step"],
        "learning_rate": cfg.lr,
        "ppo_clip": cfg.clip_eps,
        "network_architecture": "separate_actor_critic",
        "network_hidden_layers": cfg.hidden_layers,
        "network_hidden_size": cfg.hidden_size,
        "minibatch_size": cfg.minibatch_size,
        "n_epochs": cfg.n_epochs,
        "rollout_length": cfg.horizon,
        "num_rollouts": cfg.rollout,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "market_sigma": cfg.sigma,
        "num_investors": cfg.num_investors,
        "order_size_mode": cfg.order_size_mode,
        "buy_probability": cfg.buy_probability,
    }
    return metrics, trajectory_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-run PPO with relative price observation")
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
        default=ROOT / "results" / "relative_price",
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
        print(f"Training relative-price seed={seed}", flush=True)
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
            f"{row['deterministic_epsilon_ask_mean']:.3f}), "
            f"saturation=({row['actor_first_layer_saturation_20480']}, "
            f"{row['actor_first_layer_saturation_60416']})",
            flush=True,
        )

    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
