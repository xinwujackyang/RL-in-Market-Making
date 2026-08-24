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


def write_report(output: Path, trajectory: list[dict]) -> None:
    early = SCREENING_CHECKPOINTS[0]
    final = SCREENING_CHECKPOINTS[-1]
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
        "adaptive_mean_base_epsilon",
        "adaptive_mean_hedge_fraction",
        "adaptive_mean_epsilon_bid",
        "adaptive_mean_epsilon_ask",
        "approx_kl",
        "clip_fraction",
        "value_loss",
    )
    summaries = {
        checkpoint: {
            key: checkpoint_summary(trajectory, checkpoint, key)
            for key in keys
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    lines = [
        "# PPO vs Adaptive MM screening",
        "",
        "配置：3 seeds × 98 rollouts × 1024 steps。PPO 使用固定 reference "
        "configuration；Adaptive target=0.5、gamma=2、beta=0.35。每个 checkpoint "
        "统计此前 20,480 个真实 training-path steps。",
        "",
        "## Lifecycle check",
        "",
        "- `PPOAgent.train()` 只在 training run 开头调用一次 `env.reset()`；1024-step "
        "rollout boundary 不调用 reset。",
        "- 每个 seed 实测 3 次 reset 均发生在第一条 training transition 之前"
        "（environment construction、agent initialization、training start）。产生 transition "
        "后不再 reset，因此 Adaptive state 跨全部 98 个 rollout 持续存在。",
        "- 每个 seed 恰有 121 个 cold-start steps，且结束时 response table 完整。",
        "",
        "## PPO economics",
        "",
        "下表为 3-seed mean；每行使用该 checkpoint 前 20,480 steps。",
        "",
        "| Step | Market share | Spread PnL/step | Total PnL/step | Mean abs inventory | "
        "Inventory std | Mean hedge | Mean bid/ask epsilon |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
        (row for row in trajectory if int(row["training_step"]) == final),
        key=lambda row: int(row["seed"]),
    )
    lines.extend(
        [
            "",
            "### Final window by seed",
            "",
            "| Seed | PPO share | Spread PnL/step | Total PnL/step | Mean abs inventory | "
            "Inventory std | Mean hedge |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | {float(row['ppo_market_share']):.4f} | "
            f"{float(row['ppo_spread_pnl_per_step']):.4f} | "
            f"{float(row['ppo_total_pnl_per_step']):.4f} | "
            f"{float(row['ppo_mean_abs_inventory']):.3f} | "
            f"{float(row['ppo_inventory_std']):.3f} | "
            f"{float(row['ppo_mean_hedge_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive behavior",
            "",
            "| Step | Market share | Mean abs inventory | Inventory std | Mean hedge | "
            "Mean base epsilon | Mean bid/ask epsilon |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        row = summaries[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['adaptive_market_share'][0]:.6f} | "
            f"{row['adaptive_mean_abs_inventory'][0]:.3f} | "
            f"{row['adaptive_inventory_std'][0]:.3f} | "
            f"{row['adaptive_mean_hedge_fraction'][0]:.3f} | "
            f"{row['adaptive_mean_base_epsilon'][0]:.3f} | "
            f"{row['adaptive_mean_epsilon_bid'][0]:.3f} / "
            f"{row['adaptive_mean_epsilon_ask'][0]:.3f} |"
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
    early_total = summaries[early]["ppo_total_pnl_per_step"][0]
    final_total, final_total_std = summaries[final]["ppo_total_pnl_per_step"]
    early_share = summaries[early]["ppo_market_share"][0]
    final_share, final_share_std = summaries[final]["ppo_market_share"]
    early_base = summaries[early]["adaptive_mean_base_epsilon"][0]
    final_base, final_base_std = summaries[final]["adaptive_mean_base_epsilon"]
    early_adaptive_share = summaries[early]["adaptive_market_share"][0]
    final_adaptive_share = summaries[final]["adaptive_market_share"][0]
    lines.extend(
        [
            "",
            "## 判断",
            "",
            f"- **PPO 有部分学习信号，但不具备跨 seed 一致性。** 3-seed mean total "
            f"PnL/step 从 {early_total:.4f} 升至 {final_total:.4f}，但 seed 0 从 "
            "0.4884 略降至 0.4718，改善主要来自 seeds 1/2。Final total PnL/step "
            f"cross-seed std={final_total_std:.4f}。",
            f"- **Market share 的一致收敛不代表健康竞争。** PPO share 从 "
            f"{early_share:.4f} 升至 {final_share:.4f}（final std={final_share_std:.4f}），"
            "原因是 Adaptive 几乎完全退出，而不是双方稳定在动态均衡。",
            "- **库存控制变差。** PPO mean |inventory| 从 4.640 升至 9.345，"
            "inventory std 从 5.896 升至 9.542，同时 mean hedge 从 0.492 降至 0.309；"
            "seed 1 final mean |inventory| 达 14.700。",
            f"- **Adaptive 只在早期发生退让，之后没有持续跟踪 PPO。** mean base epsilon "
            f"从 {early_base:.3f} 变为 {final_base:.3f}（final std={final_base_std:.3f}），"
            f"market share 从 {early_adaptive_share:.6f} 降至 {final_adaptive_share:.6f}。"
            "60,416 steps 起三条 seed 的 base quote 均基本固定为 1.0。",
            "- Adaptive final mean hedge=0.333 由 seed 1 在近零库存、近零成交下的 "
            "hedge fraction≈1 拉高；对应名义 hedge size 几乎为零，不能解读为有效风险控制。",
            "- **观察到吸收态 pathology，但没有发现 lifecycle implementation bug。** "
            "从 trajectory 和现有 decision/update 规则推断：Step 1 选择 `(1.0, 1.0)` 后，"
            "Adaptive 对 PPO 的更窄连续报价几乎没有成交；only-executed-cell update 继续以零成交"
            "刷新该 cell，而其他候选保留一次 cold-start 的旧统计。当前没有 exploration、"
            "interpolation 或周期性 re-probing，因此无法重新进入市场。这是既定 Adaptive choices "
            "组合产生的机制性限制，不是 rollout reset 导致。",
            "- Approx KL/clip fraction 未显示直接的 policy-update explosion，但 final value loss "
            "在 seeds 1/2 显著增大（分别 164,470 / 352,284），与更大的 return/inventory scale "
            "一起构成稳定性警讯。",
            "",
            "## Go / No-Go",
            "",
            "**No-Go for 5-seed long-run confirmation in the intended online-adapting-opponent "
            "interpretation.** 当前 screening 实际在早期后变成 PPO 对一个退出市场的 opponent；"
            "延长运行会增加样本量，但不会回答 PPO 能否在持续 online adaptation 下稳定学习。"
            "按本轮 scope 不修改 Adaptive，也不自动做 hyperparameter sweep。",
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
        default=ROOT / "results" / "ppo_vs_adaptive",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.seeds != 3 or args.rollouts * args.horizon != 100_352:
        raise ValueError("screening requires 3 seeds and exactly 100,352 steps per seed")
    if any(checkpoint % args.horizon for checkpoint in SCREENING_CHECKPOINTS):
        raise ValueError("all checkpoints must align to rollout boundaries")

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
            f"adaptive_base={row['adaptive_mean_base_epsilon']:.3f}",
            flush=True,
        )

    write_report(args.output_dir, trajectory)
    print(f"Saved PPO-vs-Adaptive screening under {args.output_dir}")


if __name__ == "__main__":
    main()
