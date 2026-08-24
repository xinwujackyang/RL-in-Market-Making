from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive import AdaptiveMarketMakerCompetitor
from config import Config
from environments import TwoDealerMarketEnv
from ppo import PPOAgent


SCREENING_CHECKPOINTS = (20_480, 60_416, 100_352)
CHECKPOINT_WINDOW = 20_480
PROBE_INTERVAL = 100
BASELINE_RESULTS = ROOT / "results" / "ppo_vs_adaptive"


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

    def __init__(self, *args, **kwargs) -> None:
        self.records: list[dict] = []
        self.reset_calls = 0
        super().__init__(*args, **kwargs)

    def reset(self) -> np.ndarray:
        self.reset_calls += 1
        return super().reset()

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

    if len(environment.records) != cfg.total_steps:
        raise RuntimeError("training recorder did not capture every environment step")
    if environment.reset_calls != 3:
        raise RuntimeError(
            f"unexpected reset count {environment.reset_calls}; rollout boundaries may reset state"
        )
    cold_start_steps = sum(row["adaptive_cold_start"] for row in environment.records)
    if cold_start_steps != 121:
        raise RuntimeError(f"expected one 121-step cold start, observed {cold_start_steps}")
    adaptive.response_table.validate_complete()
    probe_steps = sum(row["adaptive_probe"] for row in environment.records)
    expected_probe_steps = (cfg.total_steps - cold_start_steps) // PROBE_INTERVAL
    if probe_steps != expected_probe_steps:
        raise RuntimeError(
            f"expected {expected_probe_steps} probe steps, observed {probe_steps}"
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


def checkpoint_summary(rows: list[dict], checkpoint: int, key: str) -> tuple[float, float]:
    values = [
        float(row[key])
        for row in rows
        if int(row["training_step"]) == checkpoint
    ]
    return float(np.mean(values)), float(np.std(values))


def write_report(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen PPO against an online Adaptive MM")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=98)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "ppo_vs_adaptive_probing",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.seeds != 3 or args.rollouts * args.horizon != 100_352:
        raise ValueError("screening requires 3 seeds and exactly 100,352 steps per seed")
    if any(checkpoint % args.horizon for checkpoint in SCREENING_CHECKPOINTS):
        raise ValueError("all checkpoints must align to rollout boundaries")
    baseline_path = BASELINE_RESULTS / "training_trajectories.csv"
    baseline_trajectory = read_csv(baseline_path)
    if len(baseline_trajectory) != args.seeds * len(SCREENING_CHECKPOINTS):
        raise RuntimeError(f"paired no-probing baseline is incomplete: {baseline_path}")

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
        print(f"Training PPO vs Adaptive seed={seed}", flush=True)
        row, trajectory_rows = train_seed(
            seed,
            args.rollouts,
            args.horizon,
            SCREENING_CHECKPOINTS,
        )
        metrics.append(row)
        trajectory.extend(trajectory_rows)
        write_csv(metrics_path, metrics)
        write_csv(trajectory_path, trajectory)
        print(
            f"Finished seed={seed}: share={row['ppo_market_share']:.3f} "
            f"total={row['ppo_total_pnl_per_step']:.3f} "
            f"adaptive_share={row['adaptive_market_share']:.3f} "
            f"adaptive_base1={row['adaptive_base_epsilon_one_frequency']:.3f}",
            flush=True,
        )

    write_report(args.output_dir, trajectory, baseline_trajectory)
    print(f"Saved PPO-vs-Adaptive screening under {args.output_dir}")


if __name__ == "__main__":
    main()
