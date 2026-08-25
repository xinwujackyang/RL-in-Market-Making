from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rlmm_phase11_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive import AdaptiveMarketMakerCompetitor
from baselines import PersistentMarketMaker
from environments import TwoDealerMarketEnv
from market import evolve_mid_price, reference_spread
from ppo_vs_adaptive import (
    CHECKPOINT_WINDOW,
    FAST_PROBE_INTERVAL,
    LONGRUN_CHECKPOINTS,
    SCREENING_CHECKPOINTS,
    RecordingAdaptiveEnv,
    initialize_and_calibrate,
    make_config,
    read_csv,
    validate_training_run,
    write_csv,
)


OUTPUT = ROOT / "results" / "ppo_vs_adaptive_pnl"
LONGRUN_OUTPUT = ROOT / "results" / "ppo_vs_adaptive_probe20_longrun"
NORMALIZED_SPREAD_OUTPUT = ROOT / "results" / "ppo_vs_adaptive_normalized_spread"
REFERENCE_TRAJECTORY = (
    ROOT
    / "results"
    / "ppo_vs_adaptive_calibrated_probe20"
    / "training_trajectories.csv"
)
SEEDS = (0, 1, 2)
ROLLOUTS = 98
HORIZON = 1024
AGENTS = ("PPO", "Adaptive")
FINAL_CHECKPOINT = SCREENING_CHECKPOINTS[-1]
FINAL_METRICS = (
    "market_share",
    "spread_pnl_per_step",
    "spread_pnl_per_captured_unit",
    "inventory_pnl_per_step",
    "hedge_cost_per_step",
    "total_pnl_per_step",
    "mean_abs_inventory",
)


class PnlRecordingEnv(RecordingAdaptiveEnv):
    """Experiment-local two-agent PnL recorder using simulator-native fields."""

    def step(self, ppo_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, info = super().step(ppo_action)
        record = self.records[-1]
        ppo_volume = float(info["gross_investor_volume"] - info["competitor_gross_volume"])
        record.update(
            ppo_inventory_pnl=float(info["inventory_pnl"]),
            ppo_hedge_cost=float(info["hedge_cost"]),
            ppo_gross_volume=ppo_volume,
            adaptive_spread_pnl=float(info["competitor_spread_pnl"]),
            adaptive_inventory_pnl=float(info["competitor_inventory_pnl"]),
            adaptive_hedge_cost=float(info["competitor_hedge_cost"]),
            adaptive_total_pnl=float(info["competitor_pnl"]),
            adaptive_gross_volume=float(info["competitor_gross_volume"]),
        )
        for prefix in ("ppo", "adaptive"):
            identity = (
                float(record[f"{prefix}_spread_pnl"])
                + float(record[f"{prefix}_inventory_pnl"])
                - float(record[f"{prefix}_hedge_cost"])
            )
            if not np.isclose(
                identity,
                float(record[f"{prefix}_total_pnl"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"{prefix} step PnL identity failed")
        return observation, reward, done, info


def summarize_agent_window(
    seed: int,
    checkpoint: int,
    agent: str,
    records: list[dict],
) -> dict:
    start = max(0, checkpoint - CHECKPOINT_WINDOW)
    window = records[start:checkpoint]
    prefix = "ppo" if agent == "PPO" else "adaptive"
    spread = np.asarray([row[f"{prefix}_spread_pnl"] for row in window], dtype=float)
    inventory_pnl = np.asarray(
        [row[f"{prefix}_inventory_pnl"] for row in window], dtype=float
    )
    hedge_cost = np.asarray(
        [row[f"{prefix}_hedge_cost"] for row in window], dtype=float
    )
    total = np.asarray([row[f"{prefix}_total_pnl"] for row in window], dtype=float)
    volume = np.asarray(
        [row[f"{prefix}_gross_volume"] for row in window], dtype=float
    )
    inventory = np.asarray(
        [row[f"{prefix}_inventory"] for row in window], dtype=float
    )
    market_share = np.asarray(
        [row[f"{prefix}_market_share"] for row in window], dtype=float
    )
    epsilon_bid = np.asarray(
        [row[f"{prefix}_epsilon_bid"] for row in window], dtype=float
    )
    epsilon_ask = np.asarray(
        [row[f"{prefix}_epsilon_ask"] for row in window], dtype=float
    )
    if not np.allclose(spread + inventory_pnl - hedge_cost, total, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{agent} aggregate source contains a PnL identity failure")
    captured_volume = float(volume.sum())
    if captured_volume <= 0.0:
        raise RuntimeError(f"{agent} captured no volume in a checkpoint window")
    return {
        "seed": seed,
        "training_step": checkpoint,
        "agent": agent,
        "window_start": start + 1,
        "window_steps": len(window),
        "market_share": float(market_share.mean()),
        "captured_volume_per_step": float(volume.mean()),
        "spread_pnl_per_step": float(spread.mean()),
        "spread_pnl_per_captured_unit": float(spread.sum() / captured_volume),
        "inventory_pnl_per_step": float(inventory_pnl.mean()),
        "hedge_cost_per_step": float(hedge_cost.mean()),
        "total_pnl_per_step": float(total.mean()),
        "mean_abs_inventory": float(np.abs(inventory).mean()),
        "mean_symmetric_quote_level": float(((epsilon_bid + epsilon_ask) / 2).mean()),
    }


def run_seed(
    seed: int,
    *,
    rollouts: int = ROLLOUTS,
    checkpoints: tuple[int, ...] = SCREENING_CHECKPOINTS,
) -> list[dict]:
    cfg = make_config(seed, rollouts, HORIZON)
    agent, calibrated_statistics, calibration = initialize_and_calibrate(cfg, seed)
    if calibration["calibration_steps"] != 1_210:
        raise RuntimeError("unexpected calibration length")
    adaptive = AdaptiveMarketMakerCompetitor(
        market_share_target=0.5,
        risk_aversion=2.0,
        probe_interval=FAST_PROBE_INTERVAL,
        initial_response_statistics=calibrated_statistics,
    )
    environment = PnlRecordingEnv(
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
    if cold_start_steps != 0 or probe_steps != cfg.total_steps // FAST_PROBE_INTERVAL:
        raise RuntimeError("formal adaptive lifecycle differs from Phase 8")
    return [
        summarize_agent_window(seed, checkpoint, agent_name, environment.records)
        for checkpoint in checkpoints
        for agent_name in AGENTS
    ]


def verify_phase8_reproduction(rows: list[dict]) -> None:
    reference = read_csv(REFERENCE_TRAJECTORY)
    if len(reference) != 9:
        raise RuntimeError(f"Phase 8 reference is incomplete: {REFERENCE_TRAJECTORY}")
    reference_by_key = {
        (int(row["seed"]), int(row["training_step"])): row for row in reference
    }
    current_by_key = {
        (int(row["seed"]), int(row["training_step"]), row["agent"]): row
        for row in rows
    }
    for seed in SEEDS:
        for checkpoint in SCREENING_CHECKPOINTS:
            expected = reference_by_key[(seed, checkpoint)]
            ppo = current_by_key[(seed, checkpoint, "PPO")]
            adaptive = current_by_key[(seed, checkpoint, "Adaptive")]
            for current, reference_key in (
                (ppo["market_share"], "ppo_market_share"),
                (ppo["total_pnl_per_step"], "ppo_total_pnl_per_step"),
                (ppo["mean_abs_inventory"], "ppo_mean_abs_inventory"),
                (adaptive["market_share"], "adaptive_market_share"),
                (adaptive["mean_abs_inventory"], "adaptive_mean_abs_inventory"),
            ):
                if not np.isclose(
                    float(current),
                    float(expected[reference_key]),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError("PnL rerun does not reproduce Phase 8 trajectory")


def metric_mean_std(rows: list[dict], agent: str, metric: str) -> tuple[float, float]:
    values = np.asarray(
        [
            float(row[metric])
            for row in rows
            if row["training_step"] == FINAL_CHECKPOINT and row["agent"] == agent
        ],
        dtype=float,
    )
    return float(values.mean()), float(values.std())


def final_statistics(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    return {
        agent: {
            metric: metric_mean_std(rows, agent, metric) for metric in FINAL_METRICS
        }
        for agent in AGENTS
    }


def plot_final_decomposition(rows: list[dict], path: Path) -> None:
    statistics = final_statistics(rows)
    components = (
        ("Spread PnL", "spread_pnl_per_step", 1.0),
        ("Inventory PnL", "inventory_pnl_per_step", 1.0),
        ("-Hedge cost", "hedge_cost_per_step", -1.0),
    )
    positions = np.arange(len(components))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.89)
    for offset, color, agent in (
        (-width / 2, "#4477AA", "PPO"),
        (width / 2, "#EE6677", "Adaptive"),
    ):
        means = np.asarray(
            [statistics[agent][metric][0] * sign for _, metric, sign in components]
        )
        errors = np.asarray(
            [statistics[agent][metric][1] for _, metric, _ in components]
        )
        bars = axis.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            color=color,
            label=agent,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set_title("Final PPO vs Adaptive PnL decomposition")
    axis.set_xlabel("PnL component")
    axis.set_ylabel("Mean PnL per environment step")
    axis.set_xticks(positions, [label for label, _, _ in components])
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_mean_std(value: tuple[float, float], digits: int = 4) -> str:
    return f"{value[0]:.{digits}f} ({value[1]:.{digits}f})"


def write_report(rows: list[dict]) -> None:
    statistics = final_statistics(rows)
    final_rows = sorted(
        (row for row in rows if row["training_step"] == FINAL_CHECKPOINT),
        key=lambda row: (int(row["seed"]), row["agent"]),
    )
    delta = {
        metric: statistics["PPO"][metric][0] - statistics["Adaptive"][metric][0]
        for metric in FINAL_METRICS
    }
    final_by_seed_agent = {
        (int(row["seed"]), row["agent"]): row for row in final_rows
    }
    ppo_per_unit_wins = sum(
        float(final_by_seed_agent[(seed, "PPO")]["spread_pnl_per_captured_unit"])
        > float(
            final_by_seed_agent[(seed, "Adaptive")]["spread_pnl_per_captured_unit"]
        )
        for seed in SEEDS
    )
    ppo_higher_inventory = sum(
        float(final_by_seed_agent[(seed, "PPO")]["mean_abs_inventory"])
        > float(final_by_seed_agent[(seed, "Adaptive")]["mean_abs_inventory"])
        for seed in SEEDS
    )
    ppo_total_wins = sum(
        float(final_by_seed_agent[(seed, "PPO")]["total_pnl_per_step"])
        > float(final_by_seed_agent[(seed, "Adaptive")]["total_pnl_per_step"])
        for seed in SEEDS
    )
    pnl_delta_identity = (
        delta["spread_pnl_per_step"]
        + delta["inventory_pnl_per_step"]
        - delta["hedge_cost_per_step"]
    )
    if not np.isclose(
        pnl_delta_identity,
        delta["total_pnl_per_step"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("final mean PnL delta identity failed")

    labels = {
        "market_share": "Market share",
        "spread_pnl_per_step": "Spread PnL/step",
        "spread_pnl_per_captured_unit": "Spread PnL/captured unit",
        "inventory_pnl_per_step": "Inventory PnL/step",
        "hedge_cost_per_step": "Hedge cost/step",
        "total_pnl_per_step": "Total PnL/step",
        "mean_abs_inventory": "Mean abs inventory",
    }
    lines = [
        "# PPO vs Adaptive MM: realized PnL decomposition",
        "",
        "## Setup and sanity",
        "",
        "Phase 10b artifacts do not retain both agents' per-step PnL components, so this analysis "
        "reran the identical Phase 8/10b benchmark: calibrated Adaptive MM, probe interval=20, "
        "current reference PPO, 3 seeds x 100,352 formal steps. Saved Phase 8 market share, PPO "
        "total PnL and both agents' mean absolute inventories reproduce to 1e-12. Every recorded "
        "step and every aggregate satisfy Spread + Inventory - HedgeCost = Total.",
        "",
        "## Evolution: 3-seed means",
        "",
        "| Step | Agent | Share | Spread/step | Spread/unit | Inventory/step | Hedge cost/step | "
        "Total/step | Mean abs inventory |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in SCREENING_CHECKPOINTS:
        for agent in AGENTS:
            selected = [
                row
                for row in rows
                if row["training_step"] == checkpoint and row["agent"] == agent
            ]
            means = {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in FINAL_METRICS
            }
            lines.append(
                f"| {checkpoint:,} | {agent} | {means['market_share']:.4f} | "
                f"{means['spread_pnl_per_step']:.4f} | "
                f"{means['spread_pnl_per_captured_unit']:.4f} | "
                f"{means['inventory_pnl_per_step']:.4f} | "
                f"{means['hedge_cost_per_step']:.4f} | "
                f"{means['total_pnl_per_step']:.4f} | "
                f"{means['mean_abs_inventory']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Final by seed",
            "",
            "| Seed | Agent | Share | Spread/step | Spread/unit | Inventory/step | Hedge cost/step | "
            "Total/step | Mean abs inventory |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | {row['agent']} | {float(row['market_share']):.4f} | "
            f"{float(row['spread_pnl_per_step']):.4f} | "
            f"{float(row['spread_pnl_per_captured_unit']):.4f} | "
            f"{float(row['inventory_pnl_per_step']):.4f} | "
            f"{float(row['hedge_cost_per_step']):.4f} | "
            f"{float(row['total_pnl_per_step']):.4f} | "
            f"{float(row['mean_abs_inventory']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Final comparison: 3-seed mean (population std)",
            "",
            "| Metric | PPO | Adaptive | PPO - Adaptive |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in FINAL_METRICS:
        lines.append(
            f"| {labels[metric]} | {render_mean_std(statistics['PPO'][metric])} | "
            f"{render_mean_std(statistics['Adaptive'][metric])} | {delta[metric]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- **Spread economics:** PPO has lower final market share "
            f"({statistics['PPO']['market_share'][0]:.4f} vs "
            f"{statistics['Adaptive']['market_share'][0]:.4f}) and therefore lower spread PnL/step "
            f"({statistics['PPO']['spread_pnl_per_step'][0]:.4f} vs "
            f"{statistics['Adaptive']['spread_pnl_per_step'][0]:.4f}). However, its spread per "
            f"captured unit is higher ({statistics['PPO']['spread_pnl_per_captured_unit'][0]:.4f} "
            f"vs {statistics['Adaptive']['spread_pnl_per_captured_unit'][0]:.4f}) in "
            f"{ppo_per_unit_wins}/3 seeds. Phase 10's quote widening therefore has a real per-unit "
            "monetization benefit, but it does not offset the volume disadvantage.",
            f"- **Inventory risk:** PPO mean absolute inventory is higher "
            f"({statistics['PPO']['mean_abs_inventory'][0]:.3f} vs "
            f"{statistics['Adaptive']['mean_abs_inventory'][0]:.3f}) in "
            f"{ppo_higher_inventory}/3 seeds. Yet PPO inventory PnL/step is lower on average "
            f"({statistics['PPO']['inventory_pnl_per_step'][0]:.4f} vs "
            f"{statistics['Adaptive']['inventory_pnl_per_step'][0]:.4f}) and changes sign across "
            "seeds. There is clear extra exposure, but no positive aggregate risk-taking explanation "
            "for PPO performance.",
            f"- **Hedging:** hedge cost/step is effectively identical "
            f"({statistics['PPO']['hedge_cost_per_step'][0]:.4f} vs "
            f"{statistics['Adaptive']['hedge_cost_per_step'][0]:.4f}). Combined with PPO's higher "
            "inventory exposure, these data do not support a hedging-efficiency advantage or reduced "
            "reliance on external hedging from cost alone.",
            f"- **Total:** PPO total PnL/step is "
            f"{statistics['PPO']['total_pnl_per_step'][0]:.4f} vs Adaptive "
            f"{statistics['Adaptive']['total_pnl_per_step'][0]:.4f}, with PPO winning only "
            f"{ppo_total_wins}/3 seed paths. The mean difference "
            f"{delta['total_pnl_per_step']:.4f} decomposes into spread "
            f"{delta['spread_pnl_per_step']:.4f} + inventory "
            f"{delta['inventory_pnl_per_step']:.4f} - hedge-cost difference "
            f"{delta['hedge_cost_per_step']:.4f}. The main realized economic advantage belongs to "
            "Adaptive and comes from greater captured flow/spread PnL per step; PPO's narrower edge "
            "is better spread monetization per captured unit.",
            "",
        ]
    )
    (OUTPUT / "report_zh.md").write_text("\n".join(lines))


def checkpoint_mean_std(
    rows: list[dict], checkpoint: int, agent: str, metric: str
) -> tuple[float, float]:
    values = np.asarray(
        [
            float(row[metric])
            for row in rows
            if row["training_step"] == checkpoint and row["agent"] == agent
        ],
        dtype=float,
    )
    return float(values.mean()), float(values.std())


def plot_longrun_economics(rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.89)
    x_values = np.asarray(LONGRUN_CHECKPOINTS, dtype=float)
    for color, marker, agent in (
        ("#4477AA", "o", "PPO"),
        ("#EE6677", "s", "Adaptive"),
    ):
        statistics = [
            checkpoint_mean_std(rows, checkpoint, agent, "total_pnl_per_step")
            for checkpoint in LONGRUN_CHECKPOINTS
        ]
        axis.errorbar(
            x_values,
            [value[0] for value in statistics],
            yerr=[value[1] for value in statistics],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            capsize=4,
            label=agent,
        )
    axis.set_title("Long-run PPO vs Adaptive total PnL")
    axis.set_xlabel("Formal training step")
    axis.set_ylabel("Mean total PnL per environment step")
    axis.set_xticks(
        x_values, [f"{checkpoint / 1_000:.0f}k" for checkpoint in LONGRUN_CHECKPOINTS]
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_longrun_report(rows: list[dict], output: Path) -> None:
    final = LONGRUN_CHECKPOINTS[-1]
    comparison_checkpoints = (100_352, 150_528, final)
    final_statistics_by_agent = {
        agent: {
            metric: checkpoint_mean_std(rows, final, agent, metric)
            for metric in FINAL_METRICS
        }
        for agent in AGENTS
    }
    delta = {
        metric: final_statistics_by_agent["PPO"][metric][0]
        - final_statistics_by_agent["Adaptive"][metric][0]
        for metric in FINAL_METRICS
    }
    final_rows = sorted(
        (row for row in rows if row["training_step"] == final),
        key=lambda row: (int(row["seed"]), row["agent"]),
    )
    final_seed_totals = {
        int(row["seed"]): {
            agent_row["agent"]: float(agent_row["total_pnl_per_step"])
            for agent_row in final_rows
            if int(agent_row["seed"]) == int(row["seed"])
        }
        for row in final_rows
    }
    ppo_total_wins = sum(
        values["PPO"] > values["Adaptive"] for values in final_seed_totals.values()
    )
    ppo_paths = {
        metric: [
            checkpoint_mean_std(rows, checkpoint, "PPO", metric)
            for checkpoint in comparison_checkpoints
        ]
        for metric in (
            "market_share",
            "spread_pnl_per_captured_unit",
            "spread_pnl_per_step",
            "total_pnl_per_step",
            "mean_symmetric_quote_level",
        )
    }
    labels = {
        "market_share": "市场份额",
        "spread_pnl_per_step": "Spread PnL/step",
        "spread_pnl_per_captured_unit": "Spread PnL/captured unit",
        "inventory_pnl_per_step": "Inventory PnL/step",
        "hedge_cost_per_step": "Hedge cost/step",
        "total_pnl_per_step": "Total PnL/step",
        "mean_abs_inventory": "平均绝对库存",
    }
    lines = [
        "# PPO vs 稳定 Adaptive MM：204,800-step confirmation",
        "",
        "## 设置与完整性检查",
        "",
        "除 formal horizon 外，Phase 11 设置保持不变：3 seeds x 200 rollouts x 1,024 = "
        "204,800 steps。每个 seed 使用 1,210 calibration steps、完整 121 cells、formal cold "
        "start=0、probe interval=20（10,240 probe steps，严格为 5%）。100,352 checkpoint "
        "在绝对误差 1e-12 内复现 Phase 8/11 指标；每一步均满足 Spread + Inventory - "
        "HedgeCost = Total。",
        "",
        "## PPO 轨迹：3-seed mean（population std）",
        "",
        "| Step | Share | Spread/unit | Spread/step | Total PnL/step | 平均绝对库存 | Mean m |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in LONGRUN_CHECKPOINTS:
        values = {
            metric: checkpoint_mean_std(rows, checkpoint, "PPO", metric)
            for metric in (
                "market_share",
                "spread_pnl_per_captured_unit",
                "spread_pnl_per_step",
                "total_pnl_per_step",
                "mean_abs_inventory",
                "mean_symmetric_quote_level",
            )
        }
        lines.append(
            f"| {checkpoint:,} | {render_mean_std(values['market_share'])} | "
            f"{render_mean_std(values['spread_pnl_per_captured_unit'])} | "
            f"{render_mean_std(values['spread_pnl_per_step'])} | "
            f"{render_mean_std(values['total_pnl_per_step'])} | "
            f"{render_mean_std(values['mean_abs_inventory'], 3)} | "
            f"{render_mean_std(values['mean_symmetric_quote_level'])} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive 轨迹：3-seed mean（population std）",
            "",
            "| Step | Share | Spread/unit | Spread/step | Total PnL/step |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in LONGRUN_CHECKPOINTS:
        values = {
            metric: checkpoint_mean_std(rows, checkpoint, "Adaptive", metric)
            for metric in (
                "market_share",
                "spread_pnl_per_captured_unit",
                "spread_pnl_per_step",
                "total_pnl_per_step",
            )
        }
        lines.append(
            f"| {checkpoint:,} | {render_mean_std(values['market_share'])} | "
            f"{render_mean_std(values['spread_pnl_per_captured_unit'])} | "
            f"{render_mean_std(values['spread_pnl_per_step'])} | "
            f"{render_mean_std(values['total_pnl_per_step'])} |"
        )
    lines.extend(
        [
            "",
            "## 204,800-step 最终比较",
            "",
            "| 指标 | PPO | Adaptive | PPO - Adaptive |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in FINAL_METRICS:
        lines.append(
            f"| {labels[metric]} | "
            f"{render_mean_std(final_statistics_by_agent['PPO'][metric])} | "
            f"{render_mean_std(final_statistics_by_agent['Adaptive'][metric])} | "
            f"{delta[metric]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 最终结果（按 seed）",
            "",
            "| Seed | Agent | 市场份额 | Total PnL/step |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | {row['agent']} | "
            f"{float(row['market_share']):.4f} | "
            f"{float(row['total_pnl_per_step']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "**分类：Case A。** 从 100,352 到 150,528 再到 204,800，PPO 市场份额为 "
            f"{ppo_paths['market_share'][0][0]:.4f} -> "
            f"{ppo_paths['market_share'][1][0]:.4f} -> {ppo_paths['market_share'][2][0]:.4f}, "
            "同时每单位 captured volume 的 spread PnL 为 "
            f"{ppo_paths['spread_pnl_per_captured_unit'][0][0]:.4f} -> "
            f"{ppo_paths['spread_pnl_per_captured_unit'][1][0]:.4f} -> "
            f"{ppo_paths['spread_pnl_per_captured_unit'][2][0]:.4f}。PPO 最终收回成交量，同时没有放弃"
            "其较高的单位成交 realized margin。",
            "",
            f"PPO total PnL/step 上升为 {ppo_paths['total_pnl_per_step'][0][0]:.4f} -> "
            f"{ppo_paths['total_pnl_per_step'][1][0]:.4f} -> "
            f"{ppo_paths['total_pnl_per_step'][2][0]:.4f}；mean symmetric quote level 上升为 "
            f"{ppo_paths['mean_symmetric_quote_level'][0][0]:.4f} -> "
            f"{ppo_paths['mean_symmetric_quote_level'][1][0]:.4f} -> "
            f"{ppo_paths['mean_symmetric_quote_level'][2][0]:.4f}。最终 checkpoint 中，PPO 在 "
            f"{ppo_total_wins}/3 条 seed 路径上取得更高 total PnL。其平均 total-PnL 优势 "
            f"{delta['total_pnl_per_step']:.4f} 分解为 spread "
            f"{delta['spread_pnl_per_step']:.4f} + inventory "
            f"{delta['inventory_pnl_per_step']:.4f} - hedge-cost 差 "
            f"{delta['hedge_cost_per_step']:.4f}；优势来自 spread/margin-volume，而不是更高的 "
            "inventory PnL 或更低的 hedging cost。",
            "",
            "**尚未出现 plateau。** 到 204,800 时，share、quote level、spread/unit 和 total PnL "
            "仍在显著变化，同时 late training 的 cross-seed dispersion 急剧扩大（PPO "
            f"total-PnL std {ppo_paths['total_pnl_per_step'][0][1]:.4f} -> "
            f"{ppo_paths['total_pnl_per_step'][1][1]:.4f} -> "
            f"{ppo_paths['total_pnl_per_step'][2][1]:.4f}）。Long-run evidence 支持更好的 PPO "
            "economics regime，但不支持 policy/economic convergence。按照 stop rule，实验在 "
            "204,800 停止，不自动继续训练。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def reconstruct_unit_reference_spread(seed: int) -> np.ndarray:
    """Reconstruct the formal run's independent price path without PPO rerunning."""
    cfg = make_config(seed, 200, HORIZON)
    if cfg.order_size_mode != "unit":
        raise RuntimeError("normalized spread analysis requires Phase 12 unit flow")
    environment = TwoDealerMarketEnv(
        PersistentMarketMaker(),
        cfg,
        seed=seed * 100_000 + 10_000,
    )
    scales = np.empty(LONGRUN_CHECKPOINTS[-1], dtype=float)
    for step in range(len(scales)):
        scales[step] = reference_spread(environment.price, 1.0, cfg)
        environment.price = evolve_mid_price(
            environment.price, cfg, environment.price_rng
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise RuntimeError("reference spread path is not finite and positive")
    return scales


def normalized_spread_rows() -> list[dict]:
    source = read_csv(LONGRUN_OUTPUT / "metrics_by_seed_checkpoint.csv")
    expected_keys = {
        (seed, checkpoint, agent)
        for seed in SEEDS
        for checkpoint in LONGRUN_CHECKPOINTS
        for agent in AGENTS
    }
    source_by_key = {
        (int(row["seed"]), int(row["training_step"]), row["agent"]): row
        for row in source
    }
    if set(source_by_key) != expected_keys:
        raise RuntimeError("Phase 12 source is incomplete")

    rows = []
    for seed in SEEDS:
        scales = reconstruct_unit_reference_spread(seed)
        for checkpoint in LONGRUN_CHECKPOINTS:
            start = checkpoint - CHECKPOINT_WINDOW
            mean_scale = float(scales[start:checkpoint].mean())
            ppo = source_by_key[(seed, checkpoint, "PPO")]
            adaptive = source_by_key[(seed, checkpoint, "Adaptive")]
            if min(
                float(ppo["captured_volume_per_step"]),
                float(adaptive["captured_volume_per_step"]),
            ) <= 0.0:
                raise RuntimeError("captured volume must be positive")
            ppo_raw = float(ppo["spread_pnl_per_captured_unit"])
            adaptive_raw = float(adaptive["spread_pnl_per_captured_unit"])
            ppo_normalized = ppo_raw / mean_scale
            adaptive_normalized = adaptive_raw / mean_scale
            row = {
                "seed": seed,
                "training_step": checkpoint,
                "window_start": start + 1,
                "window_steps": CHECKPOINT_WINDOW,
                "mean_reference_spread_s_ref_1": mean_scale,
                "ppo_market_share": float(ppo["market_share"]),
                "adaptive_market_share": float(adaptive["market_share"]),
                "ppo_raw_spread_per_unit": ppo_raw,
                "adaptive_raw_spread_per_unit": adaptive_raw,
                "ppo_normalized_spread_monetization": ppo_normalized,
                "adaptive_normalized_spread_monetization": adaptive_normalized,
                "ppo_adaptive_normalized_ratio": (
                    ppo_normalized / adaptive_normalized
                ),
            }
            if not all(np.isfinite(float(value)) for value in row.values()):
                raise RuntimeError("normalized spread result is not finite")
            rows.append(row)
    return rows


def write_normalized_spread_report(rows: list[dict]) -> None:
    def values(checkpoint: int, key: str) -> np.ndarray:
        return np.asarray(
            [float(row[key]) for row in rows if row["training_step"] == checkpoint],
            dtype=float,
        )

    lines = [
        "# Phase 12b：Normalized Spread Monetization Sanity Check",
        "",
        "## 定义与数据边界",
        "",
        "本分析无需重新训练。使用 simulator-native unit-size reference spread "
        "`S_ref,t(1) = reference_spread(P_t, 1, cfg)`；formal environment 的 price RNG 独立于 "
        "order/routing RNG，因此可用原 seed 精确重建每一步的 reference-spread path。当前参数下 "
        "`S_ref,t(1) = (2.0 + 0.2) x 1e-4 x P_t = 2.2e-4 P_t`。",
        "",
        "Phase 12 artifact 没有保存逐步 dealer captured volume，因此不能无 rerun 地精确恢复 "
        "`sum_t V_dealer,t S_ref,t(1)`。本轮固定使用唯一的 market-scale normalization："
        "`normalized = (window aggregate spread PnL / captured volume) / window mean S_ref,t(1)`。"
        "在 unit flow 下，每一步全部 20 个 market units 共享该 `S_ref,t(1)`；但该结果不冒充 "
        "dealer-volume-weighted estimator。",
        "",
        "## 3-seed trajectory：mean（population std）",
        "",
        "| Step | Mean S_ref(1) | PPO raw/unit | Adaptive raw/unit | PPO normalized | Adaptive normalized | PPO/Adaptive ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in LONGRUN_CHECKPOINTS:
        scale = values(checkpoint, "mean_reference_spread_s_ref_1")
        ppo_raw = values(checkpoint, "ppo_raw_spread_per_unit")
        adaptive_raw = values(checkpoint, "adaptive_raw_spread_per_unit")
        ppo_normalized = values(
            checkpoint, "ppo_normalized_spread_monetization"
        )
        adaptive_normalized = values(
            checkpoint, "adaptive_normalized_spread_monetization"
        )
        ratio = float(ppo_normalized.mean() / adaptive_normalized.mean())
        lines.append(
            f"| {checkpoint:,} | {render_mean_std((float(scale.mean()), float(scale.std())), 6)} | "
            f"{render_mean_std((float(ppo_raw.mean()), float(ppo_raw.std())))} | "
            f"{render_mean_std((float(adaptive_raw.mean()), float(adaptive_raw.std())))} | "
            f"{render_mean_std((float(ppo_normalized.mean()), float(ppo_normalized.std())))} | "
            f"{render_mean_std((float(adaptive_normalized.mean()), float(adaptive_normalized.std())))} | "
            f"{ratio:.4f} |"
        )

    final_rows = sorted(
        (row for row in rows if row["training_step"] == LONGRUN_CHECKPOINTS[-1]),
        key=lambda row: int(row["seed"]),
    )
    lines.extend(
        [
            "",
            "## Final 204,800：按 seed",
            "",
            "| Seed | PPO normalized | Adaptive normalized | Ratio |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in final_rows:
        lines.append(
            f"| {int(row['seed'])} | "
            f"{float(row['ppo_normalized_spread_monetization']):.4f} | "
            f"{float(row['adaptive_normalized_spread_monetization']):.4f} | "
            f"{float(row['ppo_adaptive_normalized_ratio']):.4f} |"
        )

    ratios = []
    for checkpoint in (100_352, 150_528, 204_800):
        ppo = values(checkpoint, "ppo_normalized_spread_monetization")
        adaptive = values(checkpoint, "adaptive_normalized_spread_monetization")
        ratios.append(float(ppo.mean() / adaptive.mean()))
    final_wins = sum(
        float(row["ppo_adaptive_normalized_ratio"]) > 1.0 for row in final_rows
    )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"**Case A。** PPO/Adaptive normalized ratio 从 100k 的 {ratios[0]:.4f} "
            f"升至 150k 的 {ratios[1]:.4f} 和 200k 的 {ratios[2]:.4f}；final PPO 在 "
            f"{final_wins}/3 seeds 上均高于 Adaptive。Phase 12 的 PPO per-unit advantage "
            "在 simulator-native market spread scale normalization 后仍成立且扩大，因此后期 "
            "raw spread/unit 上升不能仅由 nominal price/reference-spread scaling 解释。",
            "",
            "按照 stop rule，本分析到此结束，不继续 PPO-vs-Adaptive diagnostics。",
            "",
        ]
    )
    (NORMALIZED_SPREAD_OUTPUT / "report_zh.md").write_text("\n".join(lines))


def run_normalized_spread_analysis() -> None:
    NORMALIZED_SPREAD_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = normalized_spread_rows()
    write_csv(
        NORMALIZED_SPREAD_OUTPUT / "normalized_spread_by_seed_checkpoint.csv", rows
    )
    write_normalized_spread_report(rows)
    print(f"Saved normalized spread analysis under {NORMALIZED_SPREAD_OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PPO-vs-Adaptive realized economics")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--longrun", action="store_true")
    modes.add_argument("--normalized-spread", action="store_true")
    args = parser.parse_args()
    if args.normalized_spread:
        run_normalized_spread_analysis()
        return
    output = LONGRUN_OUTPUT if args.longrun else OUTPUT
    rollouts = 200 if args.longrun else ROLLOUTS
    checkpoints = LONGRUN_CHECKPOINTS if args.longrun else SCREENING_CHECKPOINTS
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        print(
            f"Decomposing PPO vs Adaptive PnL seed={seed} longrun={args.longrun}",
            flush=True,
        )
        rows.extend(run_seed(seed, rollouts=rollouts, checkpoints=checkpoints))
        print(f"Finished seed={seed}", flush=True)
    verify_phase8_reproduction(rows)
    if not all(
        np.isfinite(float(row[metric]))
        for row in rows
        for metric in (*FINAL_METRICS, "mean_symmetric_quote_level")
    ):
        raise RuntimeError("PnL summary contains NaN or infinite values")
    if args.longrun:
        write_csv(output / "metrics_by_seed_checkpoint.csv", rows)
        plot_longrun_economics(rows, output / "longrun_economics.png")
        write_longrun_report(rows, output)
    else:
        write_csv(output / "pnl_by_seed_checkpoint.csv", rows)
        plot_final_decomposition(rows, output / "pnl_decomposition.png")
        write_report(rows)
    print(f"Saved PPO-vs-Adaptive PnL analysis under {output}")


if __name__ == "__main__":
    main()
