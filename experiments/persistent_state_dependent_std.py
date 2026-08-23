from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import read_csv, rollout_checkpoints, write_csv

sys.path.insert(0, str(ROOT / "src"))

from baselines import PersistentMarketMaker
from config import Config, seed_everything
from environments import TwoDealerMarketEnv
from evaluation import evaluate_policy
from ppo import PPOAgent


CONDITIONS = (False, True)


def make_env_factory(cfg: Config, seed: int, stream_offset: int = 0):
    def env_factory(episode: int) -> TwoDealerMarketEnv:
        return TwoDealerMarketEnv(
            PersistentMarketMaker(0.5, 0.5, 0.0),
            cfg,
            seed=seed * 100_000 + stream_offset + 10_000 + episode,
            reward_mode="total",
        )

    return env_factory


def train_seed(
    seed: int,
    state_dependent_std: bool,
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
        state_dependent_std=state_dependent_std,
        eval_episodes=eval_episodes,
        eval_days=eval_days,
    )
    checkpoints = rollout_checkpoints(cfg.horizon, cfg.total_steps)
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
                "state_dependent_std": cfg.state_dependent_std,
                "training_step": step,
                "deterministic_epsilon_bid_mean": evaluation["mean_epsilon_bid"],
                "deterministic_epsilon_ask_mean": evaluation["mean_epsilon_ask"],
                "deterministic_epsilon_bid_state_std": evaluation["epsilon_bid_std"],
                "deterministic_epsilon_ask_state_std": evaluation["epsilon_ask_std"],
                "latent_sigma_bid_mean": evaluation["latent_std_bid"],
                "latent_sigma_ask_mean": evaluation["latent_std_ask"],
                "latent_sigma_bid_state_std": evaluation["latent_std_bid_state_std"],
                "latent_sigma_ask_state_std": evaluation["latent_std_ask_state_std"],
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
        "seed": seed,
        "state_dependent_std": cfg.state_dependent_std,
        "training_steps": cfg.total_steps,
        "deterministic_epsilon_bid_mean": deterministic_metrics["mean_epsilon_bid"],
        "deterministic_epsilon_ask_mean": deterministic_metrics["mean_epsilon_ask"],
        "sampled_epsilon_bid_mean": sampled_metrics["mean_epsilon_bid"],
        "sampled_epsilon_ask_mean": sampled_metrics["mean_epsilon_ask"],
        "sampled_epsilon_bid_std": sampled_metrics["epsilon_bid_std"],
        "sampled_epsilon_ask_std": sampled_metrics["epsilon_ask_std"],
        "latent_sigma_bid_mean": sampled_metrics["latent_std_bid"],
        "latent_sigma_ask_mean": sampled_metrics["latent_std_ask"],
        "latent_sigma_bid_state_std": sampled_metrics["latent_std_bid_state_std"],
        "latent_sigma_ask_state_std": sampled_metrics["latent_std_ask_state_std"],
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
        "gae_value_target": cfg.gae_value_target,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "vf_clip_param": cfg.vf_clip_param,
        "kl_coef": cfg.kl_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "market_sigma": cfg.sigma,
        "num_investors": cfg.num_investors,
        "order_size_mode": cfg.order_size_mode,
        "buy_probability": cfg.buy_probability,
        "competitor_epsilon_bid": 0.5,
        "competitor_epsilon_ask": 0.5,
        "competitor_hedge_fraction": 0.0,
    }
    return metrics, trajectory_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent competitor validation of state-dependent policy std"
    )
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
        default=ROOT / "results" / "persistent_state_dependent_std",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_by_seed.csv"
    trajectory_path = args.output_dir / "training_trajectories.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    trajectory = read_csv(trajectory_path) if args.resume else []
    completed = {
        (row["state_dependent_std"] == "True", int(row["seed"])) for row in metrics
    }

    for state_dependent_std in CONDITIONS:
        label = "state-dependent" if state_dependent_std else "global"
        for seed in range(args.seeds):
            if (state_dependent_std, seed) in completed:
                print(f"Skipping completed {label} seed={seed}", flush=True)
                continue
            print(f"Training {label}-std Persistent seed={seed}", flush=True)
            row, trajectory_rows = train_seed(
                seed,
                state_dependent_std,
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
                f"Finished {label} seed={seed}: det eps="
                f"({row['deterministic_epsilon_bid_mean']:.3f}, "
                f"{row['deterministic_epsilon_ask_mean']:.3f}), market share="
                f"{row['market_share']:.3f}",
                flush=True,
            )

    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
