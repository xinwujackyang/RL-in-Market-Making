from __future__ import annotations

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


OUTPUT = ROOT / "results" / "ppo_vs_adaptive_pnl"
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
    }


def run_seed(seed: int) -> list[dict]:
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
    if cold_start_steps != 0 or probe_steps != 5_017:
        raise RuntimeError("formal adaptive lifecycle differs from Phase 8")
    return [
        summarize_agent_window(seed, checkpoint, agent_name, environment.records)
        for checkpoint in SCREENING_CHECKPOINTS
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        print(f"Decomposing PPO vs Adaptive PnL seed={seed}", flush=True)
        rows.extend(run_seed(seed))
        print(f"Finished seed={seed}", flush=True)
    verify_phase8_reproduction(rows)
    if not all(
        np.isfinite(float(row[metric]))
        for row in rows
        for metric in FINAL_METRICS
    ):
        raise RuntimeError("PnL summary contains NaN or infinite values")
    write_csv(OUTPUT / "pnl_by_seed_checkpoint.csv", rows)
    plot_final_decomposition(rows, OUTPUT / "pnl_decomposition.png")
    write_report(rows)
    print(f"Saved PPO-vs-Adaptive PnL analysis under {OUTPUT}")


if __name__ == "__main__":
    main()
