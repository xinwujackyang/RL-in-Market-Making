from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import gymnasium
import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.rllib.utils.spaces.space_utils import unsquash_action


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from long_run_random import make_env_factory, read_csv, write_csv

sys.path.insert(0, str(ROOT / "src"))

from config import Config, seed_everything
from evaluation import evaluate_policy
from rllib_env import RLlibMarketEnv


TARGET_CHECKPOINTS = (20_000, 60_000, 100_000, 150_000, 200_000)


def checkpoints(horizon: int, total_steps: int) -> set[int]:
    result = {
        min(total_steps, max(horizon, round(target / horizon) * horizon))
        for target in TARGET_CHECKPOINTS
    }
    result.add(total_steps)
    return result


class RLlibPolicyAdapter:
    def __init__(self, algorithm) -> None:
        self.module = algorithm.get_module()
        self.action_space = self.module.action_space

    def _action(self, observation: np.ndarray, explore: bool) -> np.ndarray:
        batch = {
            Columns.OBS: torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        }
        with torch.no_grad():
            if explore:
                output = self.module.forward_exploration(batch)
                distribution_class = self.module.get_exploration_action_dist_cls()
            else:
                output = self.module.forward_inference(batch)
                distribution_class = self.module.get_inference_action_dist_cls()
            distribution = distribution_class.from_logits(
                output[Columns.ACTION_DIST_INPUTS]
            )
            if not explore:
                distribution = distribution.to_deterministic()
            normalized_action = distribution.sample()[0].detach().cpu().numpy()
        action = np.asarray(
            unsquash_action(normalized_action, self.action_space), dtype=np.float32
        )
        if not self.action_space.contains(action):
            raise RuntimeError(f"RLlib produced out-of-bounds action: {action}")
        return action

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        return self._action(observation, explore=False)

    def sample_action(self, observation: np.ndarray) -> np.ndarray:
        return self._action(observation, explore=True)

    def policy_entropy(self, observation: np.ndarray) -> float:
        return 0.0


def make_config(seed: int, horizon: int) -> tuple[Config, PPOConfig]:
    market_cfg = Config(
        seed=seed,
        relative_price=True,
        include_fill_feedback=False,
        paper_pnl_observation=False,
        observation_mode="default",
        num_investors=20,
        order_size_mode="unit",
        buy_probability=0.5,
        mu=0.0,
        sigma=0.2,
    )
    model_config = DefaultModelConfig(
        fcnet_hiddens=[256, 256],
        fcnet_activation="tanh",
        vf_share_layers=False,
    )
    rllib_config = (
        PPOConfig()
        .environment(
            RLlibMarketEnv,
            env_config={"market_config": asdict(market_cfg), "seed": seed},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
            rollout_fragment_length=horizon,
            batch_mode="truncate_episodes",
        )
        .learners(num_learners=0)
        .resources(num_gpus=0)
        .training(
            lr=5e-5,
            gamma=0.999,
            lambda_=0.95,
            clip_param=0.3,
            train_batch_size_per_learner=horizon,
            minibatch_size=256,
            num_epochs=10,
            entropy_coeff=0.003,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
        )
        .rl_module(model_config=model_config)
        .debugging(seed=seed, log_level="ERROR", log_sys_usage=False)
    )
    return market_cfg, rllib_config


def train_seed(
    seed: int,
    total_steps: int,
    horizon: int,
    eval_episodes: int,
    eval_days: int,
    trajectory_eval_episodes: int,
    trajectory_eval_days: int,
) -> tuple[dict, list[dict]]:
    market_cfg, rllib_config = make_config(seed, horizon)
    algorithm = rllib_config.build_algo()
    policy = RLlibPolicyAdapter(algorithm)
    expected_checkpoints = checkpoints(horizon, total_steps)
    trajectory_factory = make_env_factory(market_cfg, seed, stream_offset=40_000)
    trajectory_steps = market_cfg.steps_per_day * trajectory_eval_days
    trajectory_rows = []

    try:
        sampled_steps = 0
        while sampled_steps < total_steps:
            result = algorithm.train()
            sampled_steps = int(result[NUM_ENV_STEPS_SAMPLED_LIFETIME])
            if sampled_steps % horizon != 0:
                raise RuntimeError(
                    f"RLlib sampled {sampled_steps} steps, not a multiple of {horizon}"
                )
            if sampled_steps > total_steps:
                raise RuntimeError(
                    f"RLlib overshot requested training budget: {sampled_steps} > {total_steps}"
                )
            if sampled_steps in expected_checkpoints:
                evaluation, _ = evaluate_policy(
                    policy,
                    trajectory_factory,
                    trajectory_eval_episodes,
                    trajectory_steps,
                    deterministic=True,
                )
                trajectory_rows.append(
                    {
                        "seed": seed,
                        "training_step": sampled_steps,
                        "deterministic_epsilon_bid_mean": evaluation["mean_epsilon_bid"],
                        "deterministic_epsilon_ask_mean": evaluation["mean_epsilon_ask"],
                    }
                )
                print(
                    f"seed={seed} step={sampled_steps}: det eps="
                    f"({evaluation['mean_epsilon_bid']:.3f}, "
                    f"{evaluation['mean_epsilon_ask']:.3f})",
                    flush=True,
                )

        if {row["training_step"] for row in trajectory_rows} != expected_checkpoints:
            raise RuntimeError("not all requested evaluation checkpoints were recorded")

        final_factory = make_env_factory(market_cfg, seed, stream_offset=80_000)
        final_steps = market_cfg.steps_per_day * eval_days
        deterministic_metrics, _ = evaluate_policy(
            policy,
            final_factory,
            eval_episodes,
            final_steps,
            deterministic=True,
        )
        seed_everything(seed * 100_000 + 90_000)
        sampled_metrics, _ = evaluate_policy(
            policy,
            final_factory,
            eval_episodes,
            final_steps,
            deterministic=False,
        )
    finally:
        algorithm.stop()

    metrics = {
        "seed": seed,
        "training_steps": total_steps,
        "deterministic_epsilon_bid_mean": deterministic_metrics["mean_epsilon_bid"],
        "deterministic_epsilon_ask_mean": deterministic_metrics["mean_epsilon_ask"],
        "deterministic_error_bid": abs(deterministic_metrics["mean_epsilon_bid"]),
        "deterministic_error_ask": abs(deterministic_metrics["mean_epsilon_ask"]),
        "sampled_epsilon_bid_mean": sampled_metrics["mean_epsilon_bid"],
        "sampled_epsilon_ask_mean": sampled_metrics["mean_epsilon_ask"],
        "sampled_epsilon_bid_std": sampled_metrics["epsilon_bid_std"],
        "sampled_epsilon_ask_std": sampled_metrics["epsilon_ask_std"],
        "market_share": sampled_metrics["mean_market_share"],
        "spread_pnl_per_step": sampled_metrics["mean_spread_pnl"],
        "inventory_std": sampled_metrics["inventory_std"],
        "total_pnl": sampled_metrics["mean_total_pnl"],
        "total_pnl_per_step": sampled_metrics["mean_total_pnl_per_step"],
        "framework": "torch",
        "network_architecture": "separate_actor_critic",
        "network_hidden_layers": 2,
        "network_hidden_size": 256,
        "learning_rate": rllib_config.lr,
        "ppo_clip": rllib_config.clip_param,
        "gamma": rllib_config.gamma,
        "gae_lambda": rllib_config.lambda_,
        "train_batch_size": rllib_config.train_batch_size_per_learner,
        "minibatch_size": rllib_config.minibatch_size,
        "n_epochs": rllib_config.num_epochs,
        "ent_coef": rllib_config.entropy_coeff,
        "vf_coef": rllib_config.vf_loss_coeff,
        "max_grad_norm": rllib_config.grad_clip,
        "rllib_kl_coeff_default": rllib_config.kl_coeff,
        "rllib_kl_target_default": rllib_config.kl_target,
        "rllib_vf_clip_default": rllib_config.vf_clip_param,
        "rllib_normalize_actions_default": rllib_config.normalize_actions,
        "rllib_clip_actions_default": rllib_config.clip_actions,
        "python_version": platform.python_version(),
        "ray_version": ray.__version__,
        "rllib_version": f"bundled_with_ray_{ray.__version__}",
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "market_sigma": market_cfg.sigma,
        "num_investors": market_cfg.num_investors,
        "order_size_mode": market_cfg.order_size_mode,
        "buy_probability": market_cfg.buy_probability,
        "observation_dimension": 5,
        "action_dimension": 3,
    }
    return metrics, trajectory_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Modern RLlib PPO reference experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=204_800)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--trajectory-eval-episodes", type=int, default=3)
    parser.add_argument("--trajectory-eval-days", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "rllib_reference",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.total_steps % args.horizon != 0:
        raise ValueError("total steps must be a multiple of horizon")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics_by_seed.csv"
    trajectory_path = args.output_dir / "training_trajectories.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    trajectory = read_csv(trajectory_path) if args.resume else []
    completed = {int(row["seed"]) for row in metrics}

    ray.init(
        include_dashboard=False,
        num_cpus=1,
        log_to_driver=False,
        _temp_dir="/tmp/rlmm-ray",
    )
    try:
        for seed in range(args.seeds):
            if seed in completed:
                print(f"Skipping completed seed={seed}", flush=True)
                continue
            print(f"Training RLlib seed={seed}", flush=True)
            row, trajectory_rows = train_seed(
                seed,
                args.total_steps,
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
                f"{row['deterministic_epsilon_ask_mean']:.3f})",
                flush=True,
            )
    finally:
        ray.shutdown()

    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
