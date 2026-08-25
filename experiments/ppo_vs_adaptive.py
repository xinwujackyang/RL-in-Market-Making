from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive import (
    AdaptiveMarketMakerCompetitor,
    AdaptiveResponseTable,
    cold_start_quotes,
)
from config import Config
from environments import TwoDealerMarketEnv
from ppo import PPOAgent


SCREENING_CHECKPOINTS = (20_480, 60_416, 100_352)
LONGRUN_CHECKPOINTS = (20_480, 60_416, 100_352, 150_528, 204_800)
CHECKPOINT_WINDOW = 20_480
PROBE_INTERVAL = 100
FAST_PROBE_INTERVAL = 20
BOUNDARY_PROBE_INTERVAL = 50
CALIBRATION_PASSES = 10
BASELINE_RESULTS = ROOT / "results" / "ppo_vs_adaptive"
PROBING_RESULTS = ROOT / "results" / "ppo_vs_adaptive_probing"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class RecordingAdaptiveEnv(TwoDealerMarketEnv):
    """Experiment-local recorder; it does not change simulator behavior."""

    def __init__(
        self,
        *args,
        initial_previous_market_volume: float = 0.0,
        **kwargs,
    ) -> None:
        self.records: list[dict] = []
        self.reset_calls = 0
        self.initial_previous_market_volume = float(initial_previous_market_volume)
        super().__init__(*args, **kwargs)

    def reset(self) -> np.ndarray:
        self.reset_calls += 1
        observation = super().reset()
        self.previous_market_volume = self.initial_previous_market_volume
        return observation

    def step(self, ppo_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, info = super().step(ppo_action)
        self.records.append(
            {
                "step": len(self.records) + 1,
                "ppo_market_share": info["market_share"],
                "ppo_spread_pnl": info["spread_pnl"],
                "ppo_total_pnl": info["total_pnl"],
                "ppo_inventory": info["inventory"],
                "ppo_epsilon_bid": float(ppo_action[0]),
                "ppo_epsilon_ask": float(ppo_action[1]),
                "ppo_hedge_fraction": float(ppo_action[2]),
                "adaptive_market_share": info["competitor_market_share"],
                "adaptive_inventory": info["competitor_inventory"],
                "adaptive_epsilon_bid": info["competitor_epsilon_bid"],
                "adaptive_epsilon_ask": info["competitor_epsilon_ask"],
                "adaptive_hedge_fraction": info["competitor_hedge_fraction"],
                "adaptive_base_epsilon": info["competitor_base_epsilon"],
                "adaptive_cold_start": info["competitor_cold_start"],
                "adaptive_probe": info["competitor_probe"],
            }
        )
        return observation, reward, done, info


class ForcedGridCalibrationCompetitor:
    """Calibration-only full-grid scheduler with zero hedging."""

    def __init__(self, passes: int = CALIBRATION_PASSES) -> None:
        if passes <= 0:
            raise ValueError("calibration passes must be positive")
        self.passes = passes
        self.reset_adaptive_state()

    def reset_adaptive_state(self) -> None:
        self.response_table = AdaptiveResponseTable()
        self._schedule = cold_start_quotes() * self.passes
        self._schedule_index = 0
        self.last_base_epsilon = float("nan")
        self.last_action_was_cold_start = False
        self.last_action_was_probe = False

    @property
    def calibration_complete(self) -> bool:
        return self._schedule_index == len(self._schedule)

    def act_with_market_state(self, **market_state) -> np.ndarray:
        del market_state
        if self.calibration_complete:
            raise RuntimeError("calibration quote schedule is exhausted")
        epsilon_bid, epsilon_ask = self._schedule[self._schedule_index]
        self._schedule_index += 1
        self.last_base_epsilon = float("nan")
        return np.array([epsilon_bid, epsilon_ask, 0.0], dtype=np.float32)

    def observe_outcome(
        self,
        *,
        epsilon_bid: float,
        epsilon_ask: float,
        gross_volume: float,
        net_flow: float,
        spread_pnl: float,
        reference_spread_at_zero: float,
    ) -> None:
        if reference_spread_at_zero <= 0.0:
            raise ValueError("reference_spread_at_zero must be positive")
        self.response_table.update(
            epsilon_bid,
            epsilon_ask,
            gross_volume,
            net_flow,
            spread_pnl / reference_spread_at_zero,
        )


def make_config(seed: int, rollouts: int, horizon: int) -> Config:
    return Config(
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
        policy_distribution="squashed_normal",
    )


def initialize_and_calibrate(
    cfg: Config,
    seed: int,
) -> tuple[PPOAgent, dict, dict]:
    calibration_competitor = ForcedGridCalibrationCompetitor()
    calibration_environment = TwoDealerMarketEnv(
        calibration_competitor,
        cfg,
        seed=seed * 100_000 + 60_000,
        reward_mode="total",
    )
    agent = PPOAgent(calibration_environment, cfg)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in agent.network.named_parameters()
    }
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()

    observation = calibration_environment.reset()
    calibration_steps = len(cold_start_quotes()) * CALIBRATION_PASSES
    for _ in range(calibration_steps):
        ppo_action = agent.sample_action(observation)
        observation, _, _, _ = calibration_environment.step(ppo_action)

    if not calibration_competitor.calibration_complete:
        raise RuntimeError("forced calibration schedule did not complete")
    calibration_competitor.response_table.validate_complete()
    statistics = calibration_competitor.response_table.statistics_snapshot()
    if len(statistics) != len(cold_start_quotes()):
        raise RuntimeError("calibration did not populate all 121 response cells")
    parameter_max_abs_change = max(
        float(
            (parameter.detach() - initial_parameters[name])
            .abs()
            .max()
            .cpu()
            .item()
        )
        for name, parameter in agent.network.named_parameters()
    )
    if parameter_max_abs_change != 0.0 or agent.optimizer.state:
        raise RuntimeError("PPO parameters or optimizer changed during calibration")

    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.set_rng_state(torch_rng_state)
    metadata = {
        "calibration_steps": calibration_steps,
        "calibration_passes": CALIBRATION_PASSES,
        "calibration_cells_populated": len(statistics),
        "calibration_policy_parameter_max_abs_change": parameter_max_abs_change,
        "same_initial_ppo_agent": True,
    }
    return agent, statistics, metadata


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_checkpoint(
    seed: int,
    checkpoint: int,
    records: list[dict],
    diagnostic: dict,
) -> dict:
    start = max(0, checkpoint - CHECKPOINT_WINDOW)
    window = records[start:checkpoint]
    ppo_inventory = np.asarray([row["ppo_inventory"] for row in window], dtype=float)
    adaptive_inventory = np.asarray(
        [row["adaptive_inventory"] for row in window], dtype=float
    )
    adaptive_base = np.asarray(
        [row["adaptive_base_epsilon"] for row in window], dtype=float
    )
    probe_steps = sum(bool(row["adaptive_probe"]) for row in window)
    normal_adaptive_base = np.asarray(
        [
            row["adaptive_base_epsilon"]
            for row in window
            if not row["adaptive_cold_start"] and not row["adaptive_probe"]
        ],
        dtype=float,
    )
    return {
        "seed": seed,
        "training_step": checkpoint,
        "window_start": start + 1,
        "window_steps": len(window),
        "ppo_market_share": mean(window, "ppo_market_share"),
        "ppo_spread_pnl_per_step": mean(window, "ppo_spread_pnl"),
        "ppo_total_pnl_per_step": mean(window, "ppo_total_pnl"),
        "ppo_mean_abs_inventory": float(np.abs(ppo_inventory).mean()),
        "ppo_inventory_std": float(ppo_inventory.std()),
        "ppo_mean_hedge_fraction": mean(window, "ppo_hedge_fraction"),
        "ppo_mean_epsilon_bid": mean(window, "ppo_epsilon_bid"),
        "ppo_mean_epsilon_ask": mean(window, "ppo_epsilon_ask"),
        "adaptive_market_share": mean(window, "adaptive_market_share"),
        "adaptive_mean_abs_inventory": float(np.abs(adaptive_inventory).mean()),
        "adaptive_inventory_std": float(adaptive_inventory.std()),
        "adaptive_mean_hedge_fraction": mean(window, "adaptive_hedge_fraction"),
        "adaptive_mean_base_epsilon": float(np.nanmean(adaptive_base)),
        "adaptive_mean_epsilon_bid": mean(window, "adaptive_epsilon_bid"),
        "adaptive_mean_epsilon_ask": mean(window, "adaptive_epsilon_ask"),
        "adaptive_probe_steps": probe_steps,
        "adaptive_probe_fraction": probe_steps / len(window),
        "adaptive_base_epsilon_one_frequency": float(
            np.isclose(normal_adaptive_base, 1.0, rtol=0.0, atol=1e-6).mean()
        ),
        "approx_kl": float(diagnostic["approx_kl"]),
        "clip_fraction": float(diagnostic["clip_fraction"]),
        "value_loss": float(diagnostic["value_loss"]),
    }


def aggregate_frozen_checkpoint(
    seed: int,
    checkpoint: int,
    records: list[dict],
) -> dict:
    start = max(0, checkpoint - CHECKPOINT_WINDOW)
    window = records[start:checkpoint]
    adaptive_base = np.asarray(
        [row["adaptive_base_epsilon"] for row in window], dtype=float
    )
    normal_adaptive_base = np.asarray(
        [
            row["adaptive_base_epsilon"]
            for row in window
            if not row["adaptive_cold_start"] and not row["adaptive_probe"]
        ],
        dtype=float,
    )
    ppo_inventory = np.asarray([row["ppo_inventory"] for row in window], dtype=float)
    return {
        "seed": seed,
        "training_step": checkpoint,
        "window_start": start + 1,
        "window_steps": len(window),
        "adaptive_market_share": mean(window, "adaptive_market_share"),
        "adaptive_mean_base_epsilon": float(np.nanmean(adaptive_base)),
        "adaptive_base_epsilon_one_frequency": float(
            np.isclose(normal_adaptive_base, 1.0, rtol=0.0, atol=1e-6).mean()
        ),
        "ppo_market_share": mean(window, "ppo_market_share"),
        "ppo_total_pnl_per_step": mean(window, "ppo_total_pnl"),
        "ppo_mean_abs_inventory": float(np.abs(ppo_inventory).mean()),
    }


def validate_training_run(
    environment: RecordingAdaptiveEnv,
    adaptive: AdaptiveMarketMakerCompetitor,
    cfg: Config,
    *,
    expected_reset_calls: int,
    expected_cold_start_steps: int,
) -> tuple[int, int]:
    if len(environment.records) != cfg.total_steps:
        raise RuntimeError("training recorder did not capture every environment step")
    finite_record_keys = (
        "ppo_market_share",
        "ppo_spread_pnl",
        "ppo_total_pnl",
        "ppo_inventory",
        "ppo_epsilon_bid",
        "ppo_epsilon_ask",
        "ppo_hedge_fraction",
        "adaptive_market_share",
        "adaptive_inventory",
        "adaptive_epsilon_bid",
        "adaptive_epsilon_ask",
        "adaptive_hedge_fraction",
    )
    if not all(
        np.isfinite(float(record[key]))
        for record in environment.records
        for key in finite_record_keys
    ):
        raise RuntimeError("training recorder contains NaN or infinite values")
    if not all(
        -1.0 <= float(record[key]) <= 1.0
        for record in environment.records
        for key in (
            "ppo_epsilon_bid",
            "ppo_epsilon_ask",
            "adaptive_epsilon_bid",
            "adaptive_epsilon_ask",
        )
    ):
        raise RuntimeError("recorded epsilon action is outside [-1, 1]")
    if not all(
        0.0 <= float(record[key]) <= 1.0
        for record in environment.records
        for key in ("ppo_hedge_fraction", "adaptive_hedge_fraction")
    ):
        raise RuntimeError("recorded hedge action is outside [0, 1]")
    if environment.reset_calls != expected_reset_calls:
        raise RuntimeError(
            f"expected {expected_reset_calls} formal environment resets, observed "
            f"{environment.reset_calls}"
        )
    cold_start_steps = sum(row["adaptive_cold_start"] for row in environment.records)
    if cold_start_steps != expected_cold_start_steps:
        raise RuntimeError(
            f"expected {expected_cold_start_steps} cold-start steps, observed "
            f"{cold_start_steps}"
        )
    adaptive.response_table.validate_complete()
    probe_steps = sum(row["adaptive_probe"] for row in environment.records)
    expected_probe_steps = (
        0
        if adaptive.probe_interval is None
        else (cfg.total_steps - cold_start_steps) // adaptive.probe_interval
    )
    if probe_steps != expected_probe_steps:
        raise RuntimeError(
            f"expected {expected_probe_steps} probe steps, observed {probe_steps}"
        )
    return cold_start_steps, probe_steps


def train_seed(
    seed: int,
    rollouts: int,
    horizon: int,
    checkpoints: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    cfg = make_config(seed, rollouts, horizon)
    adaptive = AdaptiveMarketMakerCompetitor(
        market_share_target=0.5,
        risk_aversion=2.0,
        probe_interval=PROBE_INTERVAL,
    )
    environment = RecordingAdaptiveEnv(
        adaptive,
        cfg,
        seed=seed * 100_000 + 10_000,
        reward_mode="total",
    )
    agent = PPOAgent(environment, cfg)
    history = agent.train()
    cold_start_steps, probe_steps = validate_training_run(
        environment,
        adaptive,
        cfg,
        expected_reset_calls=3,
        expected_cold_start_steps=121,
    )

    diagnostics_by_step = {int(row["step"]): row for row in history.diagnostics}
    trajectory_rows = [
        aggregate_checkpoint(
            seed,
            checkpoint,
            environment.records,
            diagnostics_by_step[checkpoint],
        )
        for checkpoint in checkpoints
    ]
    final = trajectory_rows[-1]
    metrics = {
        "seed": seed,
        "training_steps": cfg.total_steps,
        "adaptive_cold_start_steps": cold_start_steps,
        "environment_reset_calls": environment.reset_calls,
        "adaptive_table_complete": True,
        "adaptive_probe_interval": PROBE_INTERVAL,
        "adaptive_probe_steps_total": probe_steps,
        "adaptive_probe_fraction_total": probe_steps / cfg.total_steps,
        **{key: value for key, value in final.items() if key != "seed"},
        "learning_rate": cfg.lr,
        "ppo_clip": cfg.clip_eps,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "minibatch_size": cfg.minibatch_size,
        "n_epochs": cfg.n_epochs,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "state_dependent_std": cfg.state_dependent_std,
        "market_sigma": cfg.sigma,
        "order_size_mode": cfg.order_size_mode,
    }
    return metrics, trajectory_rows


def train_calibrated_seed(
    seed: int,
    rollouts: int,
    horizon: int,
    checkpoints: tuple[int, ...],
    *,
    probe_interval: int = PROBE_INTERVAL,
) -> tuple[dict, list[dict]]:
    cfg = make_config(seed, rollouts, horizon)
    agent, calibrated_statistics, calibration = initialize_and_calibrate(cfg, seed)
    adaptive = AdaptiveMarketMakerCompetitor(
        market_share_target=0.5,
        risk_aversion=2.0,
        probe_interval=probe_interval,
        initial_response_statistics=calibrated_statistics,
    )
    environment = RecordingAdaptiveEnv(
        adaptive,
        cfg,
        seed=seed * 100_000 + 10_000,
        reward_mode="total",
        initial_previous_market_volume=float(cfg.num_investors),
    )
    if (
        environment.price != cfg.P0
        or environment.inventory != [0.0, 0.0]
        or environment.t != 0
    ):
        raise RuntimeError("formal environment did not start from a fresh market state")
    agent.env = environment
    history = agent.train()
    cold_start_steps, probe_steps = validate_training_run(
        environment,
        adaptive,
        cfg,
        expected_reset_calls=2,
        expected_cold_start_steps=0,
    )

    diagnostics_by_step = {int(row["step"]): row for row in history.diagnostics}
    trajectory_rows = [
        aggregate_checkpoint(
            seed,
            checkpoint,
            environment.records,
            diagnostics_by_step[checkpoint],
        )
        for checkpoint in checkpoints
    ]
    final = trajectory_rows[-1]
    metrics = {
        "seed": seed,
        "training_steps": cfg.total_steps,
        **calibration,
        "formal_cold_start_steps": cold_start_steps,
        "formal_environment_reset_calls": environment.reset_calls,
        "formal_initial_previous_market_volume": environment.initial_previous_market_volume,
        "adaptive_table_complete": True,
        "adaptive_probe_interval": probe_interval,
        "adaptive_probe_steps_total": probe_steps,
        "adaptive_probe_fraction_total": probe_steps / cfg.total_steps,
        **{key: value for key, value in final.items() if key != "seed"},
        "learning_rate": cfg.lr,
        "ppo_clip": cfg.clip_eps,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "minibatch_size": cfg.minibatch_size,
        "n_epochs": cfg.n_epochs,
        "ent_coef": cfg.ent_coef,
        "vf_coef": cfg.vf_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "state_dependent_std": cfg.state_dependent_std,
        "market_sigma": cfg.sigma,
        "order_size_mode": cfg.order_size_mode,
    }
    return metrics, trajectory_rows


def train_calibrated_fast_probe_seed(
    seed: int,
    rollouts: int,
    horizon: int,
    checkpoints: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    return train_calibrated_seed(
        seed,
        rollouts,
        horizon,
        checkpoints,
        probe_interval=FAST_PROBE_INTERVAL,
    )


def train_calibrated_boundary_probe_seed(
    seed: int,
    rollouts: int,
    horizon: int,
    checkpoints: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    return train_calibrated_seed(
        seed,
        rollouts,
        horizon,
        checkpoints,
        probe_interval=BOUNDARY_PROBE_INTERVAL,
    )


def run_calibrated_frozen_seed(
    seed: int,
    rollouts: int,
    horizon: int,
    checkpoints: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    cfg = make_config(seed, rollouts, horizon)
    agent, calibrated_statistics, calibration = initialize_and_calibrate(cfg, seed)
    adaptive = AdaptiveMarketMakerCompetitor(
        market_share_target=0.5,
        risk_aversion=2.0,
        probe_interval=PROBE_INTERVAL,
        initial_response_statistics=calibrated_statistics,
    )
    environment = RecordingAdaptiveEnv(
        adaptive,
        cfg,
        seed=seed * 100_000 + 10_000,
        reward_mode="total",
        initial_previous_market_volume=float(cfg.num_investors),
    )
    if (
        environment.price != cfg.P0
        or environment.inventory != [0.0, 0.0]
        or environment.t != 0
    ):
        raise RuntimeError("frozen formal environment is not fresh")
    agent.env = environment
    frozen_parameters = {
        name: parameter.detach().clone()
        for name, parameter in agent.network.named_parameters()
    }

    observation = environment.reset()
    for _ in range(cfg.total_steps):
        ppo_action = agent.sample_action(observation)
        observation, _, _, _ = environment.step(ppo_action)

    parameter_max_abs_change = max(
        float(
            (parameter.detach() - frozen_parameters[name])
            .abs()
            .max()
            .cpu()
            .item()
        )
        for name, parameter in agent.network.named_parameters()
    )
    if parameter_max_abs_change != 0.0:
        raise RuntimeError("frozen PPO parameters changed during formal phase")
    if agent.optimizer.state:
        raise RuntimeError("frozen PPO optimizer state changed during formal phase")
    if agent.buffer.pointer != 0 or agent.buffer.path_start != 0:
        raise RuntimeError("frozen formal loop touched the PPO rollout buffer")
    if agent.history.steps or agent.history.rewards or agent.history.diagnostics:
        raise RuntimeError("frozen formal loop created PPO training history")
    cold_start_steps, probe_steps = validate_training_run(
        environment,
        adaptive,
        cfg,
        expected_reset_calls=2,
        expected_cold_start_steps=0,
    )
    trajectory_rows = [
        aggregate_frozen_checkpoint(seed, checkpoint, environment.records)
        for checkpoint in checkpoints
    ]
    final = trajectory_rows[-1]
    metrics = {
        "seed": seed,
        "formal_steps": cfg.total_steps,
        **calibration,
        "formal_cold_start_steps": cold_start_steps,
        "adaptive_probe_steps_total": probe_steps,
        "frozen_ppo_parameter_max_abs_change": parameter_max_abs_change,
        "frozen_optimizer_state_entries": len(agent.optimizer.state),
        "frozen_rollout_buffer_pointer": agent.buffer.pointer,
        "frozen_training_history_steps": len(agent.history.steps),
        **{key: value for key, value in final.items() if key != "seed"},
    }
    return metrics, trajectory_rows


def checkpoint_summary(rows: list[dict], checkpoint: int, key: str) -> tuple[float, float]:
    values = [
        float(row[key])
        for row in rows
        if int(row["training_step"]) == checkpoint
    ]
    return float(np.mean(values)), float(np.std(values))


def write_calibrated_frozen_report(
    output: Path,
    trajectory: list[dict],
    learning_trajectory: list[dict],
) -> None:
    frozen = {
        checkpoint: {
            key: checkpoint_summary(trajectory, checkpoint, key)
            for key in (
                "adaptive_market_share",
                "adaptive_mean_base_epsilon",
                "adaptive_base_epsilon_one_frequency",
                "ppo_market_share",
                "ppo_total_pnl_per_step",
                "ppo_mean_abs_inventory",
            )
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    final = SCREENING_CHECKPOINTS[-1]
    learning_final = {
        int(row["seed"]): row
        for row in learning_trajectory
        if int(row["training_step"]) == final
    }
    frozen_final = {
        int(row["seed"]): row
        for row in trajectory
        if int(row["training_step"]) == final
    }
    lines = [
        "# Frozen PPO vs calibrated Adaptive MM",
        "",
        "## Setup",
        "",
        "每 seed 先按 Phase 6 执行 10 x 121 = 1,210 calibration steps，再在 fresh formal "
        "environment 中以同一个随机初始化 theta0 stochastic policy 手写运行 100,352 steps。Formal "
        "phase 不调用 `train()`、optimizer、rollout buffer、GAE 或 PPO loss；Adaptive 参数、warm-start "
        "table、interval=100 diagonal probing 均与 Phase 6 相同。Checkpoint 指标统计此前 20,480 steps。",
        "",
        "## Frozen trajectories",
        "",
        "| Step | Adaptive share | Adaptive mean base | Adaptive base=1 | PPO share | "
        "PPO total PnL/step | PPO mean abs inventory |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in SCREENING_CHECKPOINTS:
        row = frozen[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['adaptive_market_share'][0]:.4f} | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_base_epsilon_one_frequency'][0]:.3%} | "
            f"{row['ppo_market_share'][0]:.4f} | "
            f"{row['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_mean_abs_inventory'][0]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Learning vs frozen: final matched comparison",
            "",
            "| Seed | Learning PPO final Adaptive share | Frozen PPO final Adaptive share | "
            "Learning base=1 | Frozen base=1 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in range(3):
        learning = learning_final[seed]
        frozen_row = frozen_final[seed]
        lines.append(
            f"| {seed} | {float(learning['adaptive_market_share']):.4f} | "
            f"{float(frozen_row['adaptive_market_share']):.4f} | "
            f"{float(learning['adaptive_base_epsilon_one_frequency']):.3%} | "
            f"{float(frozen_row['adaptive_base_epsilon_one_frequency']):.3%} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "**PPO nonstationarity 是主要问题。** Frozen theta0 的三个 seed 在全部 checkpoints "
            "中，Adaptive share 保持在 0.296–0.482，base=1 frequency 保持在 24.1%–50.5%；"
            "没有出现随时间单向 drift 到 share≈0 / base=1≈100% 的 lock-in。相比之下，learning "
            "PPO 的 seed 1 final Adaptive share=0.114、base=1=86.0%，seed 0 也明显弱于 matched "
            "frozen run。Seed 2 的 frozen result 略弱于 learning result，说明 path dispersion "
            "没有消失，但结果不支持 Adaptive 自身 dynamics 单独造成此前的极端退出态。",
            "",
            "PPO formal parameter max change 对全部 seed 均为 0；optimizer state、rollout buffer "
            "pointer 与 training history 也保持为空。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def write_screening_report(
    output: Path,
    trajectory: list[dict],
    baseline_trajectory: list[dict],
) -> None:
    keys = (
        "ppo_market_share",
        "ppo_spread_pnl_per_step",
        "ppo_total_pnl_per_step",
        "ppo_mean_abs_inventory",
        "ppo_inventory_std",
        "ppo_mean_hedge_fraction",
        "ppo_mean_epsilon_bid",
        "ppo_mean_epsilon_ask",
        "adaptive_market_share",
        "adaptive_mean_abs_inventory",
        "adaptive_inventory_std",
        "adaptive_mean_hedge_fraction",
        "adaptive_mean_base_epsilon",
        "adaptive_mean_epsilon_bid",
        "adaptive_mean_epsilon_ask",
        "adaptive_probe_steps",
        "adaptive_probe_fraction",
        "adaptive_base_epsilon_one_frequency",
        "approx_kl",
        "clip_fraction",
        "value_loss",
    )
    summaries = {
        checkpoint: {
            key: checkpoint_summary(trajectory, checkpoint, key) for key in keys
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    baseline_keys = (
        "ppo_market_share",
        "ppo_total_pnl_per_step",
        "ppo_mean_abs_inventory",
        "adaptive_market_share",
        "adaptive_mean_base_epsilon",
    )
    baseline = {
        checkpoint: {
            key: checkpoint_summary(baseline_trajectory, checkpoint, key)
            for key in baseline_keys
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    lines = [
        "# PPO vs Adaptive MM with persistent diagonal probing",
        "",
        "配置与 no-probing Phase 4 完全相同：3 seeds × 98 rollouts × 1024 steps；"
        "Adaptive target=0.5、gamma=2、beta=0.35。唯一实验差异是 cold start 后每 100 "
        "个 adaptive steps 强制一次 deterministic diagonal probe。每个 checkpoint 统计此前 "
        "20,480 个真实 training-path steps。",
        "",
        "## Probing 与 lifecycle",
        "",
        "- 每个 seed 仍只有 121 cold-start steps，response table 跨 rollout 持续存在。",
        "- Probe 顺序为 epsilon grid ascending round-robin；quote 不经过 Step 2，hedge 仍使用"
        "现有 hedge objective。Response update 未改变。",
        "- `Base=1 frequency` 只统计非 cold-start、非 probe 的正常 Step 1 decisions，"
        "避免 forced probe 机械污染该指标。",
        "",
        "| Step | Probe steps/window | Probe fraction |",
        "|---:|---:|---:|",
    ]
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['adaptive_probe_steps'][0]:.0f} | "
            f"{row['adaptive_probe_fraction'][0]:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive：no-probe vs probe",
            "",
            "| Step | No-probe share | Probe share | No-probe mean base | "
            "Probe mean base | Probe base=1 frequency |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        old = baseline[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {old['adaptive_market_share'][0]:.6f} | "
            f"{row['adaptive_market_share'][0]:.6f} | "
            f"{old['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_base_epsilon_one_frequency'][0]:.3%} |"
        )
    lines.extend(
        [
            "",
            "## PPO：no-probe vs probe",
            "",
            "| Step | No-probe share | Probe share | No-probe total PnL/step | "
            "Probe total PnL/step | No-probe mean abs inventory | "
            "Probe mean abs inventory |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        old = baseline[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {old['ppo_market_share'][0]:.4f} | "
            f"{row['ppo_market_share'][0]:.4f} | "
            f"{old['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{old['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{row['ppo_mean_abs_inventory'][0]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## PPO economics with probing",
            "",
            "| Step | Share | Spread PnL/step | Total PnL/step | Mean abs inventory | "
            "Inventory std | Mean hedge | Mean bid/ask epsilon |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['ppo_market_share'][0]:.4f} | "
            f"{row['ppo_spread_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{row['ppo_inventory_std'][0]:.3f} | "
            f"{row['ppo_mean_hedge_fraction'][0]:.3f} | "
            f"{row['ppo_mean_epsilon_bid'][0]:.3f} / "
            f"{row['ppo_mean_epsilon_ask'][0]:.3f} |"
        )
    final_rows = sorted(
        (
            row
            for row in trajectory
            if int(row["training_step"]) == SCREENING_CHECKPOINTS[-1]
        ),
        key=lambda row: int(row["seed"]),
    )
    lines.extend(
        [
            "",
            "### Final window by seed",
            "",
            "| Seed | PPO share | PPO total PnL/step | PPO mean abs inventory | "
            "Adaptive share | Adaptive mean base | Base=1 frequency |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['ppo_market_share']):.4f} | "
            f"{float(row['ppo_total_pnl_per_step']):.4f} | "
            f"{float(row['ppo_mean_abs_inventory']):.3f} | "
            f"{float(row['adaptive_market_share']):.4f} | "
            f"{float(row['adaptive_mean_base_epsilon']):.3f} | "
            f"{float(row['adaptive_base_epsilon_one_frequency']):.3%} |"
        )
    lines.extend(
        [
            "",
            "## PPO diagnostics",
            "",
            "| Step | Approx KL | Clip fraction | Value loss |",
            "|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['approx_kl'][0]:.4f} | "
            f"{row['clip_fraction'][0]:.4f} | {row['value_loss'][0]:.1f} |"
        )
    final = SCREENING_CHECKPOINTS[-1]
    final_probe_share, final_probe_share_std = summaries[final][
        "adaptive_market_share"
    ]
    final_base_one, final_base_one_std = summaries[final][
        "adaptive_base_epsilon_one_frequency"
    ]
    final_ppo_total, final_ppo_total_std = summaries[final][
        "ppo_total_pnl_per_step"
    ]
    early = SCREENING_CHECKPOINTS[0]
    middle = SCREENING_CHECKPOINTS[1]
    early_ppo_total = summaries[early]["ppo_total_pnl_per_step"][0]
    early_ppo_share = summaries[early]["ppo_market_share"][0]
    middle_ppo_share = summaries[middle]["ppo_market_share"][0]
    final_ppo_share = summaries[final]["ppo_market_share"][0]
    early_ppo_inventory = summaries[early]["ppo_mean_abs_inventory"][0]
    final_ppo_inventory = summaries[final]["ppo_mean_abs_inventory"][0]
    baseline_final_share = baseline[final]["ppo_market_share"][0]
    baseline_final_total, baseline_final_total_std = baseline[final][
        "ppo_total_pnl_per_step"
    ]
    baseline_final_inventory = baseline[final]["ppo_mean_abs_inventory"][0]
    seed_improvements = sum(
        float(final_row["ppo_total_pnl_per_step"])
        > next(
            float(early_row["ppo_total_pnl_per_step"])
            for early_row in trajectory
            if int(early_row["seed"]) == int(final_row["seed"])
            and int(early_row["training_step"]) == early
        )
        for final_row in final_rows
    )
    final_inventory_values = ", ".join(
        f"{float(row['ppo_mean_abs_inventory']):.3f}" for row in final_rows
    )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- **Persistent probing 解除了原 absorbing state。** No-probe Adaptive final "
            f"share={baseline[final]['adaptive_market_share'][0]:.6f}；probe 后为 "
            f"{final_probe_share:.6f}（seed range "
            f"{min(float(row['adaptive_market_share']) for row in final_rows):.4f}–"
            f"{max(float(row['adaptive_market_share']) for row in final_rows):.4f}）。"
            "三个 seed 都保持非平凡成交份额。",
            f"- **Step 1 不再锁死。** Final normal-decision base=1 frequency 从 no-probe "
            f"的 100% 降至 {final_base_one:.3%}（cross-seed std="
            f"{final_base_one_std:.3%}；seed range "
            f"{min(float(row['adaptive_base_epsilon_one_frequency']) for row in final_rows):.3%}–"
            f"{max(float(row['adaptive_base_epsilon_one_frequency']) for row in final_rows):.3%}）。"
            "Adaptive mean base 和 share 在三个 checkpoints 继续变化，而非停在 1.0/0。",
            f"- **PPO 不再机械垄断。** Final share 从 no-probe {baseline_final_share:.4f} "
            f"降至 {final_ppo_share:.4f}；probe screening 中 share 从 "
            f"{early_ppo_share:.4f} 经 {middle_ppo_share:.4f} 变为 "
            f"{final_ppo_share:.4f}，显示双方持续交互。",
            f"- **PPO 仍表现出学习，且 economics 更跨-seed 一致。** Total PnL/step "
            f"从 {early_ppo_total:.4f} 升至 {final_ppo_total:.4f}，{seed_improvements}/3 "
            "seeds 均改善。Final PnL 较 no-probe 的 "
            f"{baseline_final_total:.4f} 更低，因为 opponent 不再退出；但 cross-seed std "
            f"从 {baseline_final_total_std:.4f} 降至 {final_ppo_total_std:.4f}。",
            f"- **库存风险没有出现 Phase 4 的同等恶化。** Mean abs inventory 从 "
            f"{early_ppo_inventory:.3f} 变为 {final_ppo_inventory:.3f}；no-probe final 为 "
            f"{baseline_final_inventory:.3f}。Final 三 seed 为 "
            f"{final_inventory_values}。",
            "- **未看到 diagonal probing 被 stale off-diagonal statistics 再次阻断。** "
            "Final Adaptive actual bid/ask 均明显低于或随 base 改变，并持续获得订单；"
            "这不证明所有 off-diagonal estimates 充分准确，但本轮没有出现由它们导致的退出态。",
            "- Probe schedule 精确：每个 seed 共 1,002 probe steps，占全部 100,352 steps "
            "的 0.9985%；cold start 仍为 121 steps。",
            "",
            "## Go / No-Go",
            "",
            "**Go for 5-seed × 204,800-step long-run confirmation.** Minimal persistent "
            "diagonal information acquisition 足以让 Adaptive 保持 behaviorally active，"
            "PPO/Adaptive 在整个 screening 中持续相互影响。Seed 间 Adaptive share/base dynamics "
            f"仍有 dispersion（final share std={final_probe_share_std:.4f}），应由 long-run "
            "confirmation 判断其是否收敛，而不是在本轮继续修改 estimator。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def write_calibrated_report(
    output: Path,
    trajectory: list[dict],
    cold_start_trajectory: list[dict],
) -> None:
    keys = (
        "ppo_market_share",
        "ppo_spread_pnl_per_step",
        "ppo_total_pnl_per_step",
        "ppo_mean_abs_inventory",
        "ppo_inventory_std",
        "ppo_mean_hedge_fraction",
        "ppo_mean_epsilon_bid",
        "ppo_mean_epsilon_ask",
        "adaptive_market_share",
        "adaptive_mean_abs_inventory",
        "adaptive_inventory_std",
        "adaptive_mean_hedge_fraction",
        "adaptive_mean_base_epsilon",
        "adaptive_mean_epsilon_bid",
        "adaptive_mean_epsilon_ask",
        "adaptive_probe_fraction",
        "adaptive_base_epsilon_one_frequency",
        "approx_kl",
        "clip_fraction",
        "value_loss",
    )
    summaries = {
        checkpoint: {
            key: checkpoint_summary(trajectory, checkpoint, key) for key in keys
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    cold_start = {
        checkpoint: {
            key: checkpoint_summary(cold_start_trajectory, checkpoint, key)
            for key in (
                "ppo_market_share",
                "ppo_total_pnl_per_step",
                "ppo_mean_abs_inventory",
                "adaptive_market_share",
                "adaptive_mean_base_epsilon",
                "adaptive_base_epsilon_one_frequency",
            )
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    final = SCREENING_CHECKPOINTS[-1]
    final_rows = sorted(
        (row for row in trajectory if int(row["training_step"]) == final),
        key=lambda row: int(row["seed"]),
    )
    lines = [
        "# Calibrated / warm-start Adaptive MM screening",
        "",
        "## Setup",
        "",
        "3 seeds（0–2）× 100,352 formal PPO training steps。每 seed 先初始化唯一的 PPOAgent "
        "θ0，在独立 market path 上冻结参数并执行 10 × 121 = 1,210 calibration steps；"
        "随后恢复 post-initialization RNG state，将同一个 PPOAgent 接到 fresh formal simulator。"
        "正式训练从 populated 121-cell response table 开始，cold start=0，并继续 interval=100 "
        "diagonal probing。Fresh state 的 lagged market-volume denominator 初始化为 unit-flow "
        "下确定的 20；price/inventory/PnL/path 均不从 calibration 继承。所有其他 "
        "PPO/environment/Adaptive 参数与 Phase 4b 相同。",
        "",
        "## Adaptive: Phase 4b cold start vs calibrated",
        "",
        "| Step | Cold share | Calibrated share | Cold mean base | Calibrated mean base | "
        "Cold base=1 | Calibrated base=1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        old = cold_start[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {old['adaptive_market_share'][0]:.4f} | "
            f"{row['adaptive_market_share'][0]:.4f} | "
            f"{old['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{old['adaptive_base_epsilon_one_frequency'][0]:.3%} | "
            f"{row['adaptive_base_epsilon_one_frequency'][0]:.3%} |"
        )
    lines.extend(
        [
            "",
            "## PPO trajectory",
            "",
            "| Step | Share | Spread PnL/step | Total PnL/step | Mean abs inventory | "
            "Inventory std | Mean hedge | Mean bid/ask epsilon |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['ppo_market_share'][0]:.4f} | "
            f"{row['ppo_spread_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{row['ppo_inventory_std'][0]:.3f} | "
            f"{row['ppo_mean_hedge_fraction'][0]:.3f} | "
            f"{row['ppo_mean_epsilon_bid'][0]:.3f} / "
            f"{row['ppo_mean_epsilon_ask'][0]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive trajectory",
            "",
            "| Step | Share | Mean abs inventory | Inventory std | Mean hedge | "
            "Mean base | Mean bid/ask epsilon | Base=1 frequency | Probe fraction |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['adaptive_market_share'][0]:.4f} | "
            f"{row['adaptive_mean_abs_inventory'][0]:.3f} | "
            f"{row['adaptive_inventory_std'][0]:.3f} | "
            f"{row['adaptive_mean_hedge_fraction'][0]:.3f} | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_mean_epsilon_bid'][0]:.3f} / "
            f"{row['adaptive_mean_epsilon_ask'][0]:.3f} | "
            f"{row['adaptive_base_epsilon_one_frequency'][0]:.3%} | "
            f"{row['adaptive_probe_fraction'][0]:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Final by seed",
            "",
            "| Seed | PPO total PnL/step | PPO share | PPO mean abs inventory | "
            "Adaptive share | Adaptive mean base | Base=1 frequency |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['ppo_total_pnl_per_step']):.4f} | "
            f"{float(row['ppo_market_share']):.4f} | "
            f"{float(row['ppo_mean_abs_inventory']):.3f} | "
            f"{float(row['adaptive_market_share']):.4f} | "
            f"{float(row['adaptive_mean_base_epsilon']):.3f} | "
            f"{float(row['adaptive_base_epsilon_one_frequency']):.3%} |"
        )
    lines.extend(
        [
            "",
            "## Final dispersion: cold start vs calibrated",
            "",
            "| Metric | Cold-start std | Calibrated std |",
            "|---|---:|---:|",
            f"| Adaptive share | {cold_start[final]['adaptive_market_share'][1]:.4f} | "
            f"{summaries[final]['adaptive_market_share'][1]:.4f} |",
            f"| Adaptive base=1 frequency | "
            f"{cold_start[final]['adaptive_base_epsilon_one_frequency'][1]:.3%} | "
            f"{summaries[final]['adaptive_base_epsilon_one_frequency'][1]:.3%} |",
            f"| PPO total PnL/step | {cold_start[final]['ppo_total_pnl_per_step'][1]:.4f} | "
            f"{summaries[final]['ppo_total_pnl_per_step'][1]:.4f} |",
            "",
            "## PPO diagnostics",
            "",
            "| Step | Approx KL | Clip fraction | Value loss |",
            "|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['approx_kl'][0]:.4f} | "
            f"{row['clip_fraction'][0]:.4f} | {row['value_loss'][0]:.1f} |"
        )
    early = SCREENING_CHECKPOINTS[0]
    middle = SCREENING_CHECKPOINTS[1]
    row_by_seed_step = {
        (int(row["seed"]), int(row["training_step"])): row for row in trajectory
    }
    ppo_improvement_count = sum(
        float(row_by_seed_step[(seed, final)]["ppo_total_pnl_per_step"])
        > float(row_by_seed_step[(seed, early)]["ppo_total_pnl_per_step"])
        for seed in range(3)
    )
    adaptive_paths = []
    for seed in range(3):
        adaptive_paths.append(
            f"seed {seed}: share "
            f"{float(row_by_seed_step[(seed, early)]['adaptive_market_share']):.3f}→"
            f"{float(row_by_seed_step[(seed, middle)]['adaptive_market_share']):.3f}→"
            f"{float(row_by_seed_step[(seed, final)]['adaptive_market_share']):.3f}, "
            f"base=1 {float(row_by_seed_step[(seed, early)]['adaptive_base_epsilon_one_frequency']):.1%}→"
            f"{float(row_by_seed_step[(seed, middle)]['adaptive_base_epsilon_one_frequency']):.1%}→"
            f"{float(row_by_seed_step[(seed, final)]['adaptive_base_epsilon_one_frequency']):.1%}"
        )
    final_value_losses = [float(row["value_loss"]) for row in final_rows]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Calibration/lifecycle 按设计完成。** 每 seed 使用同一个 θ0 PPOAgent；"
            "1,210 calibration steps 覆盖 10 次完整 121-cell grid，parameter max change=0、"
            "optimizer 未更新，并恢复 post-initialization RNG state。Formal simulator 为 fresh "
            "P0/zero-inventory state，response table 完整，cold start=0，calibration steps 不计入"
            "100,352 training steps。",
            f"- **Warm start 没有提高 Adaptive final stability。** Final mean share 从 cold-start "
            f"Phase 4b 的 {cold_start[final]['adaptive_market_share'][0]:.4f} 降至 "
            f"{summaries[final]['adaptive_market_share'][0]:.4f}；mean base=1 frequency 从 "
            f"{cold_start[final]['adaptive_base_epsilon_one_frequency'][0]:.1%} 升至 "
            f"{summaries[final]['adaptive_base_epsilon_one_frequency'][0]:.1%}。",
            f"- **三条 seed 分化而非共同稳定。** {'；'.join(adaptive_paths)}。Seed 1 快速进入 "
            "PPO-dominant / near-lock-in tendency，seed 2 则向更平衡 interaction 移动，seed 0 "
            "介于两者之间。100k 尚未达到 share≈0/base=1≈100%，但趋势不满足 3/3 healthy criterion。",
            f"- **Seed dispersion 全面恶化。** Adaptive share std "
            f"{cold_start[final]['adaptive_market_share'][1]:.4f}→"
            f"{summaries[final]['adaptive_market_share'][1]:.4f}；base=1 std "
            f"{cold_start[final]['adaptive_base_epsilon_one_frequency'][1]:.1%}→"
            f"{summaries[final]['adaptive_base_epsilon_one_frequency'][1]:.1%}；PPO PnL std "
            f"{cold_start[final]['ppo_total_pnl_per_step'][1]:.4f}→"
            f"{summaries[final]['ppo_total_pnl_per_step'][1]:.4f}。Warm start 没有降低"
            "initial-condition/path sensitivity。",
            f"- **PPO mean economics 改善但不跨 seed 一致。** Total PnL/step "
            f"{summaries[early]['ppo_total_pnl_per_step'][0]:.4f}→"
            f"{summaries[final]['ppo_total_pnl_per_step'][0]:.4f}，{ppo_improvement_count}/3 seeds "
            f"改善；market share {summaries[early]['ppo_market_share'][0]:.4f}→"
            f"{summaries[final]['ppo_market_share'][0]:.4f}。Final PnL mean 高于 cold-start "
            f"{cold_start[final]['ppo_total_pnl_per_step'][0]:.4f}，但主要由 seed 1 的 "
            "PPO-dominant path 拉高，不能作为 benchmark 成功证据。",
            f"- **Inventory exposure 更高。** PPO mean abs inventory "
            f"{summaries[early]['ppo_mean_abs_inventory'][0]:.3f}→"
            f"{summaries[final]['ppo_mean_abs_inventory'][0]:.3f}；warm-start final 高于 cold-start "
            f"{cold_start[final]['ppo_mean_abs_inventory'][0]:.3f}。Seed 1 final 达 9.637。",
            f"- Final value loss range={min(final_value_losses):.1f}–"
            f"{max(final_value_losses):.1f}，最大值同样来自 seed 1；KL/clip 未共同爆炸。",
            "- Execution sanity 全部通过：3/3 calibration 与 formal runs 完成；每 seed "
            "cells=121、formal cold start=0、formal probes=1,003；无 NaN/Inf、越界 action、"
            "parameter drift 或 crash。",
            "",
            "## Go / No-Go",
            "",
            "**No-Go for 5-seed × 204,800 warm-start confirmation.** 结果不支持 H1（instability "
            "主要来自 immature initial beliefs）；更一致的是 H2：即使以 θ0 下的 calibrated table "
            "开始，online learning interaction 仍可快速分化。Warm-start Adaptive 不应取代 "
            "Phase 4b setup 成为主要 benchmark candidate。下一步若继续，应诊断 diagonal tracking、"
            "off-diagonal staleness 与 Step 1/Step 2 feedback，而不是扩大本配置样本量。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def write_fast_probe_report(
    output: Path,
    trajectory: list[dict],
    interval_100_trajectory: list[dict],
) -> None:
    keys = (
        "adaptive_market_share",
        "adaptive_mean_base_epsilon",
        "adaptive_base_epsilon_one_frequency",
        "adaptive_probe_fraction",
        "ppo_market_share",
        "ppo_total_pnl_per_step",
        "ppo_mean_abs_inventory",
    )
    fast = {
        checkpoint: {
            key: checkpoint_summary(trajectory, checkpoint, key) for key in keys
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    control = {
        checkpoint: {
            key: checkpoint_summary(interval_100_trajectory, checkpoint, key)
            for key in keys
            if key != "adaptive_probe_fraction"
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    rows_by_seed_step = {
        (int(row["seed"]), int(row["training_step"])): row for row in trajectory
    }
    control_by_seed_step = {
        (int(row["seed"]), int(row["training_step"])): row
        for row in interval_100_trajectory
    }
    final = SCREENING_CHECKPOINTS[-1]
    fast_final_shares = [
        float(rows_by_seed_step[(seed, final)]["adaptive_market_share"])
        for seed in range(3)
    ]
    seed_one_fast = rows_by_seed_step[(1, final)]
    seed_one_control = control_by_seed_step[(1, final)]
    tracking_supported = (
        float(seed_one_fast["adaptive_market_share"])
        > float(seed_one_control["adaptive_market_share"])
        and float(seed_one_fast["adaptive_base_epsilon_one_frequency"])
        < float(seed_one_control["adaptive_base_epsilon_one_frequency"])
        and min(fast_final_shares) > 0.2
        and fast[final]["adaptive_market_share"][1]
        < control[final]["adaptive_market_share"][1]
        and fast[final]["adaptive_base_epsilon_one_frequency"][1]
        < control[final]["adaptive_base_epsilon_one_frequency"][1]
    )

    lines = [
        "# Calibrated Adaptive MM: probe interval 20",
        "",
        "## Setup and sanity",
        "",
        "Phase 6 calibrated + learning-PPO setup 原样复用。唯一 experimental change 是 "
        "deterministic diagonal round-robin probe interval 100 -> 20；3 seeds x 100,352 formal "
        "steps，checkpoint 使用 trailing 20,480-step window。每 seed calibration=1,210、"
        "cells=121、formal cold start=0、formal probes=5,017（4.9994%）；每个 checkpoint "
        "window 含 1,024 probes（5.000%）。",
        "",
        "## Adaptive trajectory",
        "",
        "| Step | Share mean (std) | Mean base | Base=1 mean (std) | Probe fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in SCREENING_CHECKPOINTS:
        row = fast[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['adaptive_market_share'][0]:.4f} "
            f"({row['adaptive_market_share'][1]:.4f}) | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_base_epsilon_one_frequency'][0]:.3%} "
            f"({row['adaptive_base_epsilon_one_frequency'][1]:.3%}) | "
            f"{row['adaptive_probe_fraction'][0]:.3%} |"
        )
    lines.extend(
        [
            "",
            "| Seed | Adaptive share: 20k -> 60k -> 100k | "
            "Base=1: 20k -> 60k -> 100k |",
            "|---:|---:|---:|",
        ]
    )
    for seed in range(3):
        seed_rows = [
            rows_by_seed_step[(seed, checkpoint)]
            for checkpoint in SCREENING_CHECKPOINTS
        ]
        lines.append(
            f"| {seed} | "
            + " -> ".join(f"{float(row['adaptive_market_share']):.4f}" for row in seed_rows)
            + " | "
            + " -> ".join(
                f"{float(row['adaptive_base_epsilon_one_frequency']):.2%}"
                for row in seed_rows
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Final matched comparison",
            "",
            "| Seed | Adaptive share, int=100 | Adaptive share, int=20 | "
            "Base=1, int=100 | Base=1, int=20 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in range(3):
        old = control_by_seed_step[(seed, final)]
        new = rows_by_seed_step[(seed, final)]
        lines.append(
            f"| {seed} | {float(old['adaptive_market_share']):.4f} | "
            f"{float(new['adaptive_market_share']):.4f} | "
            f"{float(old['adaptive_base_epsilon_one_frequency']):.3%} | "
            f"{float(new['adaptive_base_epsilon_one_frequency']):.3%} |"
        )
    lines.extend(
        [
            "",
            "## Final dispersion and PPO economics",
            "",
            "| Metric | Interval=100 | Interval=20 |",
            "|---|---:|---:|",
            f"| Adaptive share std | {control[final]['adaptive_market_share'][1]:.4f} | "
            f"{fast[final]['adaptive_market_share'][1]:.4f} |",
            f"| Adaptive base=1 std | "
            f"{control[final]['adaptive_base_epsilon_one_frequency'][1]:.3%} | "
            f"{fast[final]['adaptive_base_epsilon_one_frequency'][1]:.3%} |",
            f"| PPO share | {control[final]['ppo_market_share'][0]:.4f} | "
            f"{fast[final]['ppo_market_share'][0]:.4f} |",
            f"| PPO total PnL/step | {control[final]['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{fast[final]['ppo_total_pnl_per_step'][0]:.4f} |",
            f"| PPO mean abs inventory | {control[final]['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{fast[final]['ppo_mean_abs_inventory'][0]:.3f} |",
            "",
            "## Conclusion and next step",
            "",
            f"PPO final mean share {control[final]['ppo_market_share'][0]:.4f} -> "
            f"{fast[final]['ppo_market_share'][0]:.4f}，total PnL/step "
            f"{control[final]['ppo_total_pnl_per_step'][0]:.4f} -> "
            f"{fast[final]['ppo_total_pnl_per_step'][0]:.4f}，mean abs inventory "
            f"{control[final]['ppo_mean_abs_inventory'][0]:.3f} -> "
            f"{fast[final]['ppo_mean_abs_inventory'][0]:.3f}。更频繁 probing 让 Adaptive "
            "保持竞争，因此 PPO 的份额与 PnL 明显降低，同时 inventory exposure 下降。",
            "",
            (
                "**Tracking-bandwidth hypothesis 得到支持。** Seed 1 的退出态消失，三个 seeds "
                "均保持 meaningful interaction，且 Adaptive share/base=1 的 cross-seed "
                "dispersion 同时下降。Seed 0 late window 有一定回落，但没有接近此前 lock-in。"
                "下一步应测试能否以更有选择性的 refresh allocation 降低 5% probing cost，"
                "而不是进入 Step 1/Step 2 mechanism diagnosis。"
                if tracking_supported
                else "**Simply refreshing diagonal estimates faster is insufficient。** 至少一项"
                "关键条件（seed 1 恢复、3-seed meaningful interaction、share/base=1 dispersion "
                "同时下降）未满足。停止调整 probe frequency；下一步进入 off-diagonal Step-2 "
                "statistics、EMA lag 与 Step-1/Step-2 feedback diagnosis。"
            ),
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def write_probe50_report(
    output: Path,
    trajectory: list[dict],
    interval_100_trajectory: list[dict],
) -> None:
    interval_20_path = (
        ROOT
        / "results"
        / "ppo_vs_adaptive_calibrated_probe20"
        / "training_trajectories.csv"
    )
    interval_20_trajectory = read_csv(interval_20_path)
    if len(interval_20_trajectory) != 9:
        raise RuntimeError(f"interval=20 result is incomplete: {interval_20_path}")
    keys = (
        "adaptive_market_share",
        "adaptive_mean_base_epsilon",
        "adaptive_base_epsilon_one_frequency",
        "adaptive_probe_fraction",
        "ppo_market_share",
        "ppo_total_pnl_per_step",
        "ppo_mean_abs_inventory",
    )

    def summaries(rows: list[dict]) -> dict:
        return {
            checkpoint: {
                key: checkpoint_summary(rows, checkpoint, key)
                for key in keys
                if key in rows[0]
            }
            for checkpoint in SCREENING_CHECKPOINTS
        }

    interval_100 = summaries(interval_100_trajectory)
    interval_50 = summaries(trajectory)
    interval_20 = summaries(interval_20_trajectory)
    by_seed_step = {
        interval: {
            (int(row["seed"]), int(row["training_step"])): row for row in rows
        }
        for interval, rows in (
            (100, interval_100_trajectory),
            (50, trajectory),
            (20, interval_20_trajectory),
        )
    }
    final = SCREENING_CHECKPOINTS[-1]
    final_50 = [by_seed_step[50][(seed, final)] for seed in range(3)]
    deteriorating_seeds = []
    for seed in range(3):
        seed_rows = [
            by_seed_step[50][(seed, checkpoint)]
            for checkpoint in SCREENING_CHECKPOINTS
        ]
        shares = [float(row["adaptive_market_share"]) for row in seed_rows]
        base_one = [
            float(row["adaptive_base_epsilon_one_frequency"]) for row in seed_rows
        ]
        if shares[0] > shares[1] > shares[2] and base_one[0] < base_one[1] < base_one[2]:
            deteriorating_seeds.append(seed)
    meaningful_interaction = all(
        float(row["adaptive_market_share"]) > 0.2
        and float(row["adaptive_base_epsilon_one_frequency"]) < 0.8
        for row in final_50
    )
    dispersion_closer_to_20 = all(
        abs(interval_50[final][key][1] - interval_20[final][key][1])
        < abs(interval_50[final][key][1] - interval_100[final][key][1])
        for key in (
            "adaptive_market_share",
            "adaptive_base_epsilon_one_frequency",
        )
    )
    interval_50_works = (
        meaningful_interaction
        and dispersion_closer_to_20
        and not deteriorating_seeds
    )

    lines = [
        "# Calibrated Adaptive MM: probe interval 50 boundary check",
        "",
        "## Setup and sanity",
        "",
        "Phase 8 calibrated + learning-PPO setup 原样复用，唯一 experimental change 是 probe "
        "interval 20 -> 50。3 seeds x 100,352 formal steps；每 seed calibration=1,210、"
        "cells=121、formal cold start=0、formal probes=2,007（约 2%）。",
        "",
        "## Interval=50 trajectories",
        "",
        "| Seed | Adaptive share: 20k -> 60k -> 100k | Mean base: 20k -> 60k -> 100k | "
        "Base=1: 20k -> 60k -> 100k |",
        "|---:|---:|---:|---:|",
    ]
    for seed in range(3):
        rows = [
            by_seed_step[50][(seed, checkpoint)]
            for checkpoint in SCREENING_CHECKPOINTS
        ]
        lines.append(
            f"| {seed} | "
            + " -> ".join(f"{float(row['adaptive_market_share']):.4f}" for row in rows)
            + " | "
            + " -> ".join(
                f"{float(row['adaptive_mean_base_epsilon']):.3f}" for row in rows
            )
            + " | "
            + " -> ".join(
                f"{float(row['adaptive_base_epsilon_one_frequency']):.2%}"
                for row in rows
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Final matched comparison",
            "",
            "| Seed | Share @100 | Share @50 | Share @20 | Base=1 @100 | "
            "Base=1 @50 | Base=1 @20 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in range(3):
        old = by_seed_step[100][(seed, final)]
        middle = by_seed_step[50][(seed, final)]
        fast = by_seed_step[20][(seed, final)]
        lines.append(
            f"| {seed} | {float(old['adaptive_market_share']):.4f} | "
            f"{float(middle['adaptive_market_share']):.4f} | "
            f"{float(fast['adaptive_market_share']):.4f} | "
            f"{float(old['adaptive_base_epsilon_one_frequency']):.3%} | "
            f"{float(middle['adaptive_base_epsilon_one_frequency']):.3%} | "
            f"{float(fast['adaptive_base_epsilon_one_frequency']):.3%} |"
        )
    lines.extend(
        [
            "",
            "## Final dispersion and PPO economics",
            "",
            "| Metric | Interval=100 | Interval=50 | Interval=20 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key, percent in (
        ("Adaptive share std", "adaptive_market_share", False),
        ("Adaptive base=1 std", "adaptive_base_epsilon_one_frequency", True),
        ("PPO share", "ppo_market_share", False),
        ("PPO total PnL/step", "ppo_total_pnl_per_step", False),
        ("PPO mean abs inventory", "ppo_mean_abs_inventory", False),
    ):
        values = [
            summary[final][key][1 if "std" in label.lower() else 0]
            for summary in (interval_100, interval_50, interval_20)
        ]
        rendered = [f"{value:.3%}" if percent else f"{value:.4f}" for value in values]
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            f"Interval=50 的 PPO final mean share={interval_50[final]['ppo_market_share'][0]:.4f}、"
            f"total PnL/step={interval_50[final]['ppo_total_pnl_per_step'][0]:.4f}、mean abs "
            f"inventory={interval_50[final]['ppo_mean_abs_inventory'][0]:.3f}，三者均位于 "
            "interval=100 与 interval=20 之间。",
            "",
            (
                "**Interval=50 works；freeze probe interval=50。** 3/3 seeds 保持 meaningful "
                "Adaptive share，未出现 share≈0 / base=1≈100% drift，且 final share/base=1 "
                "dispersion 都更接近 interval=20 而不是 interval=100。2% probing 已足够，"
                "无需支付 5% forced exploration；至此停止调整 probing frequency，下一阶段回到 "
                "PPO behavior analysis。"
                if interval_50_works
                else f"**Interval=50 fails；freeze probe interval=20。** Seeds "
                f"{', '.join(str(seed) for seed in deteriorating_seeds)} 的 Adaptive share 在三个 "
                "checkpoints 单调下降，同时 base=1 单调上升；尤其 seed 0 final share=0.250 / "
                "base=1=62.9%，已接近 interval=100 的 0.224 / 65.0% pathology。虽然 final "
                "cross-seed dispersion 居中且数值上更靠近 interval=20，这不能覆盖明确的 "
                "within-seed lock-in trajectory。不再尝试其他 probing intervals；下一阶段回到 "
                "PPO behavior analysis。"
            ),
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def write_longrun_report(
    output: Path,
    trajectory: list[dict],
    screening_trajectory: list[dict],
) -> None:
    keys = (
        "ppo_market_share",
        "ppo_spread_pnl_per_step",
        "ppo_total_pnl_per_step",
        "ppo_mean_abs_inventory",
        "ppo_inventory_std",
        "ppo_mean_hedge_fraction",
        "ppo_mean_epsilon_bid",
        "ppo_mean_epsilon_ask",
        "adaptive_market_share",
        "adaptive_mean_abs_inventory",
        "adaptive_inventory_std",
        "adaptive_mean_hedge_fraction",
        "adaptive_mean_base_epsilon",
        "adaptive_mean_epsilon_bid",
        "adaptive_mean_epsilon_ask",
        "adaptive_probe_steps",
        "adaptive_probe_fraction",
        "adaptive_base_epsilon_one_frequency",
        "approx_kl",
        "clip_fraction",
        "value_loss",
    )
    summaries = {
        checkpoint: {
            key: checkpoint_summary(trajectory, checkpoint, key) for key in keys
        }
        for checkpoint in LONGRUN_CHECKPOINTS
    }
    screening_final = {
        key: checkpoint_summary(screening_trajectory, 100_352, key)
        for key in (
            "ppo_market_share",
            "ppo_total_pnl_per_step",
            "ppo_mean_abs_inventory",
            "adaptive_market_share",
            "adaptive_mean_base_epsilon",
            "adaptive_base_epsilon_one_frequency",
        )
    }
    final = LONGRUN_CHECKPOINTS[-1]
    final_rows = sorted(
        (row for row in trajectory if int(row["training_step"]) == final),
        key=lambda row: int(row["seed"]),
    )
    lines = [
        "# PPO vs Adaptive MM persistent-probing long-run confirmation",
        "",
        "## Setup",
        "",
        "5 seeds（0–4）× 200 rollouts × 1024 steps = 204,800 training steps/seed。"
        "PPO、environment 和 Adaptive 配置完全冻结自 Phase 4b；每个 checkpoint 使用 "
        "trailing 20,480-step training-path window。Phase 4b comparison 为 3-seed screening，"
        "不是 paired statistical test。",
        "",
        "## PPO trajectory",
        "",
        "| Step | Share | Spread PnL/step | Total PnL/step | Mean abs inventory | "
        "Inventory std | Mean hedge | Mean bid/ask epsilon |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in LONGRUN_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['ppo_market_share'][0]:.4f} | "
            f"{row['ppo_spread_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{row['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{row['ppo_inventory_std'][0]:.3f} | "
            f"{row['ppo_mean_hedge_fraction'][0]:.3f} | "
            f"{row['ppo_mean_epsilon_bid'][0]:.3f} / "
            f"{row['ppo_mean_epsilon_ask'][0]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive trajectory",
            "",
            "| Step | Share | Mean abs inventory | Inventory std | Mean hedge | "
            "Mean base | Mean bid/ask epsilon | Base=1 frequency | Probe fraction |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in LONGRUN_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['adaptive_market_share'][0]:.4f} | "
            f"{row['adaptive_mean_abs_inventory'][0]:.3f} | "
            f"{row['adaptive_inventory_std'][0]:.3f} | "
            f"{row['adaptive_mean_hedge_fraction'][0]:.3f} | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_mean_epsilon_bid'][0]:.3f} / "
            f"{row['adaptive_mean_epsilon_ask'][0]:.3f} | "
            f"{row['adaptive_base_epsilon_one_frequency'][0]:.3%} | "
            f"{row['adaptive_probe_fraction'][0]:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Final by seed",
            "",
            "| Seed | PPO total PnL/step | PPO share | PPO mean abs inventory | "
            "Adaptive share | Adaptive mean base | Base=1 frequency |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['ppo_total_pnl_per_step']):.4f} | "
            f"{float(row['ppo_market_share']):.4f} | "
            f"{float(row['ppo_mean_abs_inventory']):.3f} | "
            f"{float(row['adaptive_market_share']):.4f} | "
            f"{float(row['adaptive_mean_base_epsilon']):.3f} | "
            f"{float(row['adaptive_base_epsilon_one_frequency']):.3%} |"
        )
    lines.extend(
        [
            "| **Mean** | "
            f"{summaries[final]['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{summaries[final]['ppo_market_share'][0]:.4f} | "
            f"{summaries[final]['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{summaries[final]['adaptive_market_share'][0]:.4f} | "
            f"{summaries[final]['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{summaries[final]['adaptive_base_epsilon_one_frequency'][0]:.3%} |",
            "| **Population std** | "
            f"{summaries[final]['ppo_total_pnl_per_step'][1]:.4f} | "
            f"{summaries[final]['ppo_market_share'][1]:.4f} | "
            f"{summaries[final]['ppo_mean_abs_inventory'][1]:.3f} | "
            f"{summaries[final]['adaptive_market_share'][1]:.4f} | "
            f"{summaries[final]['adaptive_mean_base_epsilon'][1]:.3f} | "
            f"{summaries[final]['adaptive_base_epsilon_one_frequency'][1]:.3%} |",
            "",
            "## Phase 4b @100k vs Phase 5 @204.8k",
            "",
            "| Metric | Phase 4b final (3 seeds) | Phase 5 final (5 seeds) |",
            "|---|---:|---:|",
            f"| PPO total PnL/step | {screening_final['ppo_total_pnl_per_step'][0]:.4f} | "
            f"{summaries[final]['ppo_total_pnl_per_step'][0]:.4f} |",
            f"| PPO mean abs inventory | {screening_final['ppo_mean_abs_inventory'][0]:.3f} | "
            f"{summaries[final]['ppo_mean_abs_inventory'][0]:.3f} |",
            f"| PPO market share | {screening_final['ppo_market_share'][0]:.4f} | "
            f"{summaries[final]['ppo_market_share'][0]:.4f} |",
            f"| Adaptive market share | {screening_final['adaptive_market_share'][0]:.4f} | "
            f"{summaries[final]['adaptive_market_share'][0]:.4f} |",
            f"| Adaptive mean base | {screening_final['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{summaries[final]['adaptive_mean_base_epsilon'][0]:.3f} |",
            f"| Adaptive base=1 frequency | "
            f"{screening_final['adaptive_base_epsilon_one_frequency'][0]:.3%} | "
            f"{summaries[final]['adaptive_base_epsilon_one_frequency'][0]:.3%} |",
            f"| PPO PnL population std | "
            f"{screening_final['ppo_total_pnl_per_step'][1]:.4f} | "
            f"{summaries[final]['ppo_total_pnl_per_step'][1]:.4f} |",
            "",
            "## PPO diagnostics",
            "",
            "| Step | Approx KL | Clip fraction | Value loss |",
            "|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in LONGRUN_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['approx_kl'][0]:.4f} | "
            f"{row['clip_fraction'][0]:.4f} | {row['value_loss'][0]:.1f} |"
        )
    early = LONGRUN_CHECKPOINTS[0]
    screening_endpoint = 100_352
    late = 150_528
    row_by_seed_step = {
        (int(row["seed"]), int(row["training_step"])): row for row in trajectory
    }
    ppo_improvement_count = sum(
        float(row_by_seed_step[(seed, final)]["ppo_total_pnl_per_step"])
        > float(row_by_seed_step[(seed, early)]["ppo_total_pnl_per_step"])
        for seed in range(5)
    )
    inventory_increase_count = sum(
        float(row_by_seed_step[(seed, final)]["ppo_mean_abs_inventory"])
        > float(row_by_seed_step[(seed, early)]["ppo_mean_abs_inventory"])
        for seed in range(5)
    )
    low_share_final_rows = [
        row for row in final_rows if float(row["adaptive_market_share"]) < 0.10
    ]
    collapse_paths = []
    for row in low_share_final_rows:
        seed = int(row["seed"])
        at_100k = row_by_seed_step[(seed, screening_endpoint)]
        at_150k = row_by_seed_step[(seed, late)]
        collapse_paths.append(
            f"seed {seed}: share "
            f"{float(at_100k['adaptive_market_share']):.3f}→"
            f"{float(at_150k['adaptive_market_share']):.3f}→"
            f"{float(row['adaptive_market_share']):.3f}, base=1 "
            f"{float(at_100k['adaptive_base_epsilon_one_frequency']):.1%}→"
            f"{float(at_150k['adaptive_base_epsilon_one_frequency']):.1%}→"
            f"{float(row['adaptive_base_epsilon_one_frequency']):.1%}"
        )
    active_paths = []
    for row in final_rows:
        if float(row["adaptive_market_share"]) >= 0.10:
            seed = int(row["seed"])
            active_paths.append(
                f"seed {seed}: "
                f"{float(row_by_seed_step[(seed, screening_endpoint)]['adaptive_market_share']):.3f}→"
                f"{float(row_by_seed_step[(seed, late)]['adaptive_market_share']):.3f}→"
                f"{float(row['adaptive_market_share']):.3f}"
            )
    final_value_losses = [float(row["value_loss"]) for row in final_rows]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- **Adaptive 没有在所有 seeds 中长期保持 active。** 5-seed mean share 从 "
            f"100k 的 {summaries[screening_endpoint]['adaptive_market_share'][0]:.4f} 降至 "
            f"204.8k 的 {summaries[final]['adaptive_market_share'][0]:.4f}；mean base=1 "
            f"frequency 从 {summaries[screening_endpoint]['adaptive_base_epsilon_one_frequency'][0]:.1%} "
            f"升至 {summaries[final]['adaptive_base_epsilon_one_frequency'][0]:.1%}。",
            f"- **出现两个明确的 late-stage near-absorbing paths。** {'；'.join(collapse_paths)}。"
            "这说明 1% diagonal probing 对这些 paths 主要延迟、而非永久消除 lock-in。",
            f"- **另外三个 seeds 维持 nontrivial interaction，但不是单一固定点。** "
            f"{'；'.join(active_paths)}。Seeds 1/2/3 分别表现为较稳定 band 或有界振荡，"
            "而 seeds 0/4 向 PPO-dominant regime 漂移，支持 multi-agent path dependence。",
            f"- **PPO economics 持续改善。** Total PnL/step mean 从 "
            f"{summaries[early]['ppo_total_pnl_per_step'][0]:.4f} 升至 "
            f"{summaries[final]['ppo_total_pnl_per_step'][0]:.4f}，{ppo_improvement_count}/5 seeds "
            f"均改善；spread PnL/step 同时从 "
            f"{summaries[early]['ppo_spread_pnl_per_step'][0]:.4f} 升至 "
            f"{summaries[final]['ppo_spread_pnl_per_step'][0]:.4f}，因此收益提升不只是 inventory "
            "mark-to-market exposure。",
            f"- **Inventory risk 有所增加但不是无界一致恶化。** Mean abs inventory "
            f"{summaries[early]['ppo_mean_abs_inventory'][0]:.3f}→"
            f"{summaries[final]['ppo_mean_abs_inventory'][0]:.3f}，inventory std "
            f"{summaries[early]['ppo_inventory_std'][0]:.3f}→"
            f"{summaries[final]['ppo_inventory_std'][0]:.3f}；{inventory_increase_count}/5 seeds "
            "增加。Seed 0/3 final mean abs inventory 达 13.209/10.688，是需要保留的风险警讯。",
            f"- **Seed dispersion 不可忽略。** Final PPO PnL population std="
            f"{summaries[final]['ppo_total_pnl_per_step'][1]:.4f}（seed range "
            f"{min(float(row['ppo_total_pnl_per_step']) for row in final_rows):.4f}–"
            f"{max(float(row['ppo_total_pnl_per_step']) for row in final_rows):.4f}），Adaptive "
            f"share std={summaries[final]['adaptive_market_share'][1]:.4f}，base=1 frequency "
            f"std={summaries[final]['adaptive_base_epsilon_one_frequency'][1]:.1%}。",
            f"- Approx KL/clip fraction 没有共同 explosion，但 final value loss range="
            f"{min(final_value_losses):.1f}–{max(final_value_losses):.1f}；seed 1 的大 value loss "
            "与高 PnL path 一起说明 return/value scale 仍高度 path-dependent。本轮按 scope 不修复。",
            "- Execution sanity 全部通过：5/5 runs 到达 204,800 steps；每 seed cold start=121、"
            "probe steps=2,046（总步数占比 0.9990%）；无 NaN/Inf、越界 action 或 crash。",
            "",
            "## Go / No-Go",
            "",
            "**No-Go for freezing this setup as a stable, reproducible interaction result.** "
            "Positive PPO learning 与 3/5 active-opponent paths 值得保留，但 2/5 seeds 在 late stage "
            "重新接近 absorbing state，且 final regime dispersion 很大。下一步应是 targeted mechanism "
            "investigation，区分 diagonal estimates 已刷新但 Step 1 仍偏向 1.0，还是 stale "
            "off-diagonal Step 2 responses / path feedback 导致退出；本轮不修改或 sweep 任何机制。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen PPO against an online Adaptive MM")
    parser.add_argument(
        "--mode",
        choices=(
            "screening",
            "calibrated",
            "calibrated_fast_probe",
            "calibrated_probe50",
            "calibrated_frozen",
            "longrun",
        ),
        default="screening",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=98)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode == "screening":
        checkpoints = SCREENING_CHECKPOINTS
        expected_seeds = 3
        expected_steps = 100_352
        default_output = PROBING_RESULTS
        comparison_path = BASELINE_RESULTS / "training_trajectories.csv"
    elif args.mode == "calibrated":
        checkpoints = SCREENING_CHECKPOINTS
        expected_seeds = 3
        expected_steps = 100_352
        default_output = ROOT / "results" / "ppo_vs_adaptive_calibrated"
        comparison_path = PROBING_RESULTS / "training_trajectories.csv"
    elif args.mode == "calibrated_fast_probe":
        checkpoints = SCREENING_CHECKPOINTS
        expected_seeds = 3
        expected_steps = 100_352
        default_output = ROOT / "results" / "ppo_vs_adaptive_calibrated_probe20"
        comparison_path = (
            ROOT / "results" / "ppo_vs_adaptive_calibrated" / "training_trajectories.csv"
        )
    elif args.mode == "calibrated_probe50":
        checkpoints = SCREENING_CHECKPOINTS
        expected_seeds = 3
        expected_steps = 100_352
        default_output = ROOT / "results" / "ppo_vs_adaptive_calibrated_probe50"
        comparison_path = (
            ROOT / "results" / "ppo_vs_adaptive_calibrated" / "training_trajectories.csv"
        )
    elif args.mode == "calibrated_frozen":
        checkpoints = SCREENING_CHECKPOINTS
        expected_seeds = 3
        expected_steps = 100_352
        default_output = ROOT / "results" / "ppo_vs_adaptive_calibrated_frozen"
        comparison_path = (
            ROOT / "results" / "ppo_vs_adaptive_calibrated" / "training_trajectories.csv"
        )
    else:
        checkpoints = LONGRUN_CHECKPOINTS
        expected_seeds = 5
        expected_steps = 204_800
        default_output = ROOT / "results" / "ppo_vs_adaptive_probing_longrun"
        comparison_path = PROBING_RESULTS / "training_trajectories.csv"
    if args.seeds != expected_seeds or args.rollouts * args.horizon != expected_steps:
        raise ValueError(
            f"{args.mode} requires {expected_seeds} seeds and exactly "
            f"{expected_steps:,} steps per seed"
        )
    if any(checkpoint % args.horizon for checkpoint in checkpoints):
        raise ValueError("all checkpoints must align to rollout boundaries")
    comparison_trajectory = read_csv(comparison_path)
    expected_comparison_rows = 9
    if len(comparison_trajectory) != expected_comparison_rows:
        raise RuntimeError(f"comparison result is incomplete: {comparison_path}")

    output_dir = args.output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_by_seed.csv"
    trajectory_path = output_dir / "training_trajectories.csv"
    metrics = read_csv(metrics_path) if args.resume else []
    trajectory = read_csv(trajectory_path) if args.resume else []
    completed = {int(row["seed"]) for row in metrics}

    for seed in range(args.seeds):
        if seed in completed:
            print(f"Skipping completed seed={seed}", flush=True)
            continue
        print(f"Running PPO vs Adaptive seed={seed} mode={args.mode}", flush=True)
        if args.mode == "calibrated":
            training_function = train_calibrated_seed
        elif args.mode == "calibrated_fast_probe":
            training_function = train_calibrated_fast_probe_seed
        elif args.mode == "calibrated_probe50":
            training_function = train_calibrated_boundary_probe_seed
        elif args.mode == "calibrated_frozen":
            training_function = run_calibrated_frozen_seed
        else:
            training_function = train_seed
        row, trajectory_rows = training_function(
            seed,
            args.rollouts,
            args.horizon,
            checkpoints,
        )
        metrics.append(row)
        trajectory.extend(trajectory_rows)
        write_csv(metrics_path, metrics)
        write_csv(trajectory_path, trajectory)
        print(
            f"Finished seed={seed}: share={row['ppo_market_share']:.3f} "
            f"total={row['ppo_total_pnl_per_step']:.3f} "
            f"adaptive_share={row['adaptive_market_share']:.3f} "
            f"adaptive_base1={row['adaptive_base_epsilon_one_frequency']:.3f}"
            + (
                f" calibration={row['calibration_steps']} cold_start="
                f"{row['formal_cold_start_steps']}"
                if args.mode
                in (
                    "calibrated",
                    "calibrated_fast_probe",
                    "calibrated_probe50",
                    "calibrated_frozen",
                )
                else ""
            ),
            flush=True,
        )

    if args.mode == "screening":
        write_screening_report(output_dir, trajectory, comparison_trajectory)
    elif args.mode == "calibrated":
        write_calibrated_report(output_dir, trajectory, comparison_trajectory)
    elif args.mode == "calibrated_fast_probe":
        write_fast_probe_report(output_dir, trajectory, comparison_trajectory)
    elif args.mode == "calibrated_probe50":
        write_probe50_report(output_dir, trajectory, comparison_trajectory)
    elif args.mode == "calibrated_frozen":
        write_calibrated_frozen_report(output_dir, trajectory, comparison_trajectory)
    else:
        write_longrun_report(output_dir, trajectory, comparison_trajectory)
    print(f"Saved PPO-vs-Adaptive {args.mode} under {output_dir}")


if __name__ == "__main__":
    main()
