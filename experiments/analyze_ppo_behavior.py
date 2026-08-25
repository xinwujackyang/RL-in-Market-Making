from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rlmm_phase10_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive import AdaptiveMarketMakerCompetitor
from ppo_vs_adaptive import (
    CHECKPOINT_WINDOW,
    FAST_PROBE_INTERVAL,
    SCREENING_CHECKPOINTS,
    RecordingAdaptiveEnv,
    initialize_and_calibrate,
    make_config,
    read_csv,
    validate_training_run,
    write_csv,
)


OUTPUT = ROOT / "results" / "ppo_behavior_probe20"
REFERENCE_TRAJECTORY = (
    ROOT
    / "results"
    / "ppo_vs_adaptive_calibrated_probe20"
    / "training_trajectories.csv"
)
SEEDS = (0, 1, 2)
ROLLOUTS = 98
HORIZON = 1024
HEDGE_BINS = (
    (0.0, 2.0, "|z| < 2"),
    (2.0, 5.0, "2 <= |z| < 5"),
    (5.0, 10.0, "5 <= |z| < 10"),
    (10.0, np.inf, "|z| >= 10"),
)
SKEW_BIN_EDGES = np.asarray(
    [-np.inf, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, np.inf]
)
CHECKPOINT_LABELS = {
    SCREENING_CHECKPOINTS[0]: "Early (20,480)",
    SCREENING_CHECKPOINTS[1]: "Mid (60,416)",
    SCREENING_CHECKPOINTS[2]: "Final (100,352)",
}


class DecisionInventoryRecordingEnv(RecordingAdaptiveEnv):
    """Record PPO inventory immediately before the corresponding action executes."""

    def step(self, ppo_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        decision_inventory = float(self.inventory[0])
        result = super().step(ppo_action)
        self.records[-1]["ppo_decision_inventory"] = decision_inventory
        return result


def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return float(y.mean()), float("nan"), float("nan")
    slope = float(np.dot(x_centered, y - y.mean()) / denominator)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    total = float(np.dot(y - y.mean(), y - y.mean()))
    r_squared = float(1.0 - np.dot(residual, residual) / total) if total else 0.0
    return intercept, slope, r_squared


def summarize_window(seed: int, checkpoint: int, records: list[dict]) -> dict:
    start = max(0, checkpoint - CHECKPOINT_WINDOW)
    window = records[start:checkpoint]
    inventory = np.asarray(
        [row["ppo_decision_inventory"] for row in window], dtype=float
    )
    epsilon_bid = np.asarray([row["ppo_epsilon_bid"] for row in window], dtype=float)
    epsilon_ask = np.asarray([row["ppo_epsilon_ask"] for row in window], dtype=float)
    skew = epsilon_bid - epsilon_ask
    _, beta_bid, _ = linear_regression(inventory, epsilon_bid)
    _, beta_ask, _ = linear_regression(inventory, epsilon_ask)
    _, beta_skew, r_squared_skew = linear_regression(inventory, skew)
    return {
        "seed": seed,
        "training_step": checkpoint,
        "window_start": start + 1,
        "window_steps": len(window),
        "mean_epsilon_bid": float(epsilon_bid.mean()),
        "mean_epsilon_ask": float(epsilon_ask.mean()),
        "mean_symmetric_quote_level": float(((epsilon_bid + epsilon_ask) / 2).mean()),
        "beta_bid": beta_bid,
        "beta_ask": beta_ask,
        "beta_skew": beta_skew,
        "r_squared_skew": r_squared_skew,
    }


def run_seed(seed: int) -> tuple[list[dict], dict[int, dict[str, np.ndarray]]]:
    cfg = make_config(seed, ROLLOUTS, HORIZON)
    agent, calibrated_statistics, calibration = initialize_and_calibrate(cfg, seed)
    if calibration["calibration_steps"] != 1_210:
        raise RuntimeError("unexpected calibration length")
    adaptive = AdaptiveMarketMakerCompetitor(
        market_share_target=0.5,
        risk_aversion=2.0,
        probe_interval=FAST_PROBE_INTERVAL,
        initial_response_statistics=calibrated_statistics,
    )
    environment = DecisionInventoryRecordingEnv(
        adaptive,
        cfg,
        seed=seed * 100_000 + 10_000,
        reward_mode="total",
        initial_previous_market_volume=float(cfg.num_investors),
    )
    agent.env = environment
    agent.train()
    cold_start_steps, probe_steps = validate_training_run(
        environment,
        adaptive,
        cfg,
        expected_reset_calls=2,
        expected_cold_start_steps=0,
    )
    if cold_start_steps != 0 or probe_steps != 5_017:
        raise RuntimeError("formal adaptive lifecycle differs from Phase 8")

    summaries = []
    pooled = {}
    for checkpoint in SCREENING_CHECKPOINTS:
        summaries.append(summarize_window(seed, checkpoint, environment.records))
        start = max(0, checkpoint - CHECKPOINT_WINDOW)
        window = environment.records[start:checkpoint]
        epsilon_bid = np.asarray(
            [row["ppo_epsilon_bid"] for row in window], dtype=float
        )
        epsilon_ask = np.asarray(
            [row["ppo_epsilon_ask"] for row in window], dtype=float
        )
        pooled[checkpoint] = {
            "inventory": np.asarray(
                [row["ppo_decision_inventory"] for row in window], dtype=float
            ),
            "skew": epsilon_bid - epsilon_ask,
            "hedge": np.asarray(
                [row["ppo_hedge_fraction"] for row in window], dtype=float
            ),
        }
    return summaries, pooled


def verify_phase8_reproduction(summary_rows: list[dict]) -> None:
    reference = read_csv(REFERENCE_TRAJECTORY)
    if len(reference) != 9:
        raise RuntimeError(f"Phase 8 reference is incomplete: {REFERENCE_TRAJECTORY}")
    reference_by_key = {
        (int(row["seed"]), int(row["training_step"])): row for row in reference
    }
    for row in summary_rows:
        expected = reference_by_key[(int(row["seed"]), int(row["training_step"]))]
        for current_key, reference_key in (
            ("mean_epsilon_bid", "ppo_mean_epsilon_bid"),
            ("mean_epsilon_ask", "ppo_mean_epsilon_ask"),
        ):
            if not np.isclose(
                float(row[current_key]),
                float(expected[reference_key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError("behavior rerun does not reproduce Phase 8 actions")


def pooled_by_checkpoint(
    seed_windows: list[dict[int, dict[str, np.ndarray]]],
) -> dict[int, dict[str, np.ndarray]]:
    return {
        checkpoint: {
            key: np.concatenate([windows[checkpoint][key] for windows in seed_windows])
            for key in ("inventory", "skew", "hedge")
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }


def build_hedge_rows(pooled: dict[int, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for checkpoint in SCREENING_CHECKPOINTS:
        abs_inventory = np.abs(pooled[checkpoint]["inventory"])
        hedge = pooled[checkpoint]["hedge"]
        for lower, upper, label in HEDGE_BINS:
            mask = (abs_inventory >= lower) & (abs_inventory < upper)
            rows.append(
                {
                    "training_step": checkpoint,
                    "inventory_bin": label,
                    "lower_bound": lower,
                    "upper_bound": "inf" if np.isinf(upper) else upper,
                    "sample_count": int(mask.sum()),
                    "mean_hedge_fraction": (
                        float(hedge[mask].mean()) if mask.any() else ""
                    ),
                }
            )
    return rows


def plot_inventory_skew(pooled: dict[int, dict[str, np.ndarray]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
    colors = ("#4477AA", "#EE6677", "#228833")
    for color, checkpoint in zip(colors, SCREENING_CHECKPOINTS):
        inventory = pooled[checkpoint]["inventory"]
        skew = pooled[checkpoint]["skew"]
        bin_index = np.searchsorted(SKEW_BIN_EDGES, inventory, side="right") - 1
        x_values = []
        y_values = []
        for index in range(len(SKEW_BIN_EDGES) - 1):
            mask = bin_index == index
            if mask.any():
                x_values.append(float(inventory[mask].mean()))
                y_values.append(float(skew[mask].mean()))
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=CHECKPOINT_LABELS[checkpoint],
        )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(0.0, color="#777777", linewidth=0.8)
    axis.set_title("PPO inventory-conditioned quote skew")
    axis.set_xlabel("Decision-time inventory z")
    axis.set_ylabel("Mean epsilon_bid - epsilon_ask")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_inventory_hedge(hedge_rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.90)
    colors = ("#4477AA", "#EE6677", "#228833")
    labels = ("< 2", "[2, 5)", "[5, 10)", ">= 10")
    positions = np.arange(len(labels))
    for color, checkpoint in zip(colors, SCREENING_CHECKPOINTS):
        rows = [row for row in hedge_rows if row["training_step"] == checkpoint]
        means = [float(row["mean_hedge_fraction"]) for row in rows]
        axis.plot(
            positions,
            means,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=CHECKPOINT_LABELS[checkpoint],
        )
    axis.set_title("PPO hedge fraction by inventory magnitude")
    axis.set_xlabel("Decision-time |inventory| bin")
    axis.set_ylabel("Mean hedge fraction")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def group_mean_std(rows: list[dict], checkpoint: int, key: str) -> tuple[float, float]:
    values = np.asarray(
        [float(row[key]) for row in rows if row["training_step"] == checkpoint]
    )
    return float(values.mean()), float(values.std())


def write_report(summary_rows: list[dict], hedge_rows: list[dict]) -> None:
    quote_summary = {
        checkpoint: {
            key: group_mean_std(summary_rows, checkpoint, key)
            for key in (
                "mean_epsilon_bid",
                "mean_epsilon_ask",
                "mean_symmetric_quote_level",
            )
        }
        for checkpoint in SCREENING_CHECKPOINTS
    }
    hedge_by_step = {
        checkpoint: [
            row for row in hedge_rows if row["training_step"] == checkpoint
        ]
        for checkpoint in SCREENING_CHECKPOINTS
    }
    sign_counts = {
        checkpoint: sum(
            float(row["beta_bid"]) > 0.0
            and float(row["beta_ask"]) < 0.0
            and float(row["beta_skew"]) > 0.0
            for row in summary_rows
            if row["training_step"] == checkpoint
        )
        for checkpoint in SCREENING_CHECKPOINTS
    }
    early = SCREENING_CHECKPOINTS[0]
    final = SCREENING_CHECKPOINTS[-1]
    final_rows = [row for row in summary_rows if row["training_step"] == final]
    positive_final_skew = sum(float(row["beta_skew"]) > 0.0 for row in final_rows)
    widened_seed_count = sum(
        float(
            next(
                row["mean_symmetric_quote_level"]
                for row in summary_rows
                if row["seed"] == seed and row["training_step"] == final
            )
        )
        > float(
            next(
                row["mean_symmetric_quote_level"]
                for row in summary_rows
                if row["seed"] == seed and row["training_step"] == early
            )
        )
        for seed in SEEDS
    )
    final_hedge_means = [
        float(row["mean_hedge_fraction"]) for row in hedge_by_step[final]
    ]
    final_hedge_increasing = all(
        right > left for left, right in zip(final_hedge_means, final_hedge_means[1:])
    )

    lines = [
        "# PPO policy behavior against calibrated Adaptive MM (probe interval 20)",
        "",
        "## Setup",
        "",
        "Phase 8 artifacts only contain checkpoint aggregates, so this analysis reran the identical "
        "3-seed x 100,352-step benchmark. Calibration, RNG restoration, fresh formal environment, "
        "PPO training and Adaptive probe interval=20 are unchanged. The rerun reproduces the saved "
        "Phase 8 mean bid/ask actions to absolute tolerance 1e-12. Inventory is captured immediately "
        "before `env.step(action)`, so every regression and bin uses decision-time inventory.",
        "",
        "## Quoting evolution",
        "",
        "Values are 3-seed mean (population std), using trailing 20,480-step windows.",
        "",
        "| Step | Mean epsilon_bid | Mean epsilon_ask | Symmetric level m |",
        "|---:|---:|---:|---:|",
    ]
    for checkpoint in SCREENING_CHECKPOINTS:
        row = quote_summary[checkpoint]
        lines.append(
            f"| {checkpoint:,} | {row['mean_epsilon_bid'][0]:.4f} "
            f"({row['mean_epsilon_bid'][1]:.4f}) | "
            f"{row['mean_epsilon_ask'][0]:.4f} ({row['mean_epsilon_ask'][1]:.4f}) | "
            f"{row['mean_symmetric_quote_level'][0]:.4f} "
            f"({row['mean_symmetric_quote_level'][1]:.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Inventory skew regressions",
            "",
            "| Seed | Step | beta_bid | beta_ask | beta_skew | R-squared skew |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {int(row['seed'])} | {int(row['training_step']):,} | "
            f"{float(row['beta_bid']):.5f} | {float(row['beta_ask']):.5f} | "
            f"{float(row['beta_skew']):.5f} | {float(row['r_squared_skew']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Full economically classical sign pattern counts "
            "(beta_bid>0, beta_ask<0, beta_skew>0): "
            + ", ".join(
                f"{checkpoint:,}={sign_counts[checkpoint]}/3"
                for checkpoint in SCREENING_CHECKPOINTS
            )
            + ".",
            "",
            "## Hedging by decision-time inventory magnitude",
            "",
            "Each cell is pooled 3-seed mean hedge fraction (sample count).",
            "",
            "| Step | |z| < 2 | 2 <= |z| < 5 | 5 <= |z| < 10 | |z| >= 10 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in SCREENING_CHECKPOINTS:
        cells = [
            f"{float(row['mean_hedge_fraction']):.4f} ({int(row['sample_count']):,})"
            for row in hedge_by_step[checkpoint]
        ]
        lines.append(f"| {checkpoint:,} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- **Quoting:** pooled symmetric level changes from "
            f"{quote_summary[early]['mean_symmetric_quote_level'][0]:.4f} early to "
            f"{quote_summary[final]['mean_symmetric_quote_level'][0]:.4f} final；"
            f"{widened_seed_count}/3 seeds 都向更高 m 移动。"
            + (
                "PPO 随训练更愿意整体 widen quotes。"
                if quote_summary[final]["mean_symmetric_quote_level"][0]
                > quote_summary[early]["mean_symmetric_quote_level"][0]
                else "PPO 随训练变得更 aggressive/tighter。"
            ),
            f"- **Inventory skew:** final beta_skew 在 {positive_final_skew}/3 seeds 中为正，"
            f"但完整 classical sign pattern 只有 {sign_counts[final]}/3。Seed 1 是 bid 上调 / "
            "ask 下调；seed 0 两侧都随 inventory 上调但 bid 更快；seed 2 两侧都下调但 ask "
            "更快。因此 PPO 稳定学到了 relative skew，却没有跨 seed 学到一致的 classical "
            "two-sided decomposition。Final R-squared 仅为 "
            f"{min(float(row['r_squared_skew']) for row in final_rows):.3f}–"
            f"{max(float(row['r_squared_skew']) for row in final_rows):.3f}，inventory 对 stochastic "
            "skew 的线性解释力有限但非零。",
            "- **Hedging:** final pooled hedge means by increasing |z| bin are "
            + " -> ".join(f"{value:.4f}" for value in final_hedge_means)
            + ". "
            + (
                "它们随 inventory magnitude 单调增加。"
                if final_hedge_increasing
                else "它们并不随 inventory magnitude 增加；极端 |z|>=10 时反而最低。"
            ),
            "- **Overall:** PPO 学到的是“整体报价逐渐变宽 + 以 relative quote skew 响应 inventory”"
            "的控制结构。Economically sensible inventory management 只得到部分支持：quote skew "
            "方向最终 3/3 合理，但两侧分解不一致；hedge head 没有表现出 exposure 越大、hedge "
            "越强的经典风险控制。",
            "",
        ]
    )
    (OUTPUT / "report_zh.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    seed_windows = []
    for seed in SEEDS:
        print(f"Analyzing Phase 8 behavior seed={seed}", flush=True)
        seed_summary, pooled = run_seed(seed)
        summary_rows.extend(seed_summary)
        seed_windows.append(pooled)
        print(f"Finished seed={seed}", flush=True)

    verify_phase8_reproduction(summary_rows)
    pooled = pooled_by_checkpoint(seed_windows)
    hedge_rows = build_hedge_rows(pooled)
    if not all(int(row["sample_count"]) > 0 for row in hedge_rows):
        raise RuntimeError("a fixed inventory hedge bin is empty")
    write_csv(OUTPUT / "behavior_by_seed_checkpoint.csv", summary_rows)
    write_csv(OUTPUT / "inventory_hedge_bins.csv", hedge_rows)
    plot_inventory_skew(pooled, OUTPUT / "inventory_skew.png")
    plot_inventory_hedge(hedge_rows, OUTPUT / "inventory_hedge.png")
    write_report(summary_rows, hedge_rows)
    print(f"Saved PPO behavior analysis under {OUTPUT}")


if __name__ == "__main__":
    main()
