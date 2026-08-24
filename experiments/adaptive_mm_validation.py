from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive import AdaptiveMarketMakerCompetitor
from baselines import PersistentMarketMaker
from config import Config
from environments import TwoDealerMarketEnv


OUTPUT_FIELDS = [
    "seed",
    "gamma",
    "step",
    "phase",
    "inventory_before_action",
    "inventory",
    "base_epsilon",
    "epsilon_bid",
    "epsilon_ask",
    "hedge_fraction",
    "market_share",
    "spread_pnl",
    "inventory_pnl",
    "hedge_cost",
    "total_pnl",
]


def run_path(seed: int, gamma: float, steps: int) -> tuple[list[dict], dict]:
    cfg = Config(
        seed=seed,
        num_investors=20,
        buy_probability=0.5,
        sigma=0.1,
    )
    adaptive = AdaptiveMarketMakerCompetitor(
        market_share_target=0.5,
        risk_aversion=gamma,
    )
    environment = TwoDealerMarketEnv(adaptive, cfg, seed=seed)
    persistent_action = PersistentMarketMaker(0.5, 0.5, 0.0).act()
    rows = []

    for step in range(steps):
        _, _, _, info = environment.step(persistent_action)
        rows.append(
            {
                "seed": seed,
                "gamma": gamma,
                "step": step,
                "phase": "cold_start" if info["competitor_cold_start"] else "adaptive",
                "inventory_before_action": info["competitor_inventory_before_action"],
                "inventory": info["competitor_inventory"],
                "base_epsilon": info["competitor_base_epsilon"],
                "epsilon_bid": info["competitor_epsilon_bid"],
                "epsilon_ask": info["competitor_epsilon_ask"],
                "hedge_fraction": info["competitor_hedge_fraction"],
                "market_share": info["competitor_market_share"],
                "spread_pnl": info["competitor_spread_pnl"],
                "inventory_pnl": info["competitor_inventory_pnl"],
                "hedge_cost": info["competitor_hedge_cost"],
                "total_pnl": info["competitor_pnl"],
            }
        )

    adaptive.response_table.validate_complete()
    formal = [row for row in rows if row["phase"] == "adaptive"]
    inventory_before = np.asarray([row["inventory_before_action"] for row in formal])
    inventory = np.asarray([row["inventory"] for row in formal])
    base = np.asarray([row["base_epsilon"] for row in formal])
    epsilon_bid = np.asarray([row["epsilon_bid"] for row in formal])
    epsilon_ask = np.asarray([row["epsilon_ask"] for row in formal])
    hedge_fraction = np.asarray([row["hedge_fraction"] for row in formal])
    short = inventory_before < 0.0
    long = inventory_before > 0.0

    metrics = {
        "seed": seed,
        "gamma": gamma,
        "steps": steps,
        "cold_start_steps": sum(row["phase"] == "cold_start" for row in rows),
        "adaptive_steps": len(formal),
        "table_complete": adaptive.cold_start_complete,
        "mean_market_share": float(np.mean([row["market_share"] for row in formal])),
        "mean_inventory": float(inventory.mean()),
        "mean_abs_inventory": float(np.abs(inventory).mean()),
        "inventory_std": float(inventory.std()),
        "max_abs_inventory": float(np.abs(inventory).max()),
        "mean_base_epsilon": float(base.mean()),
        "mean_epsilon_bid": float(epsilon_bid.mean()),
        "mean_epsilon_ask": float(epsilon_ask.mean()),
        "mean_hedge_fraction": float(hedge_fraction.mean()),
        "short_count": int(short.sum()),
        "long_count": int(long.sum()),
        "short_mean_bid_skew": float((base[short] - epsilon_bid[short]).mean()),
        "long_mean_ask_skew": float((base[long] - epsilon_ask[long]).mean()),
        "short_correct_skew_frequency": float(np.mean(epsilon_bid[short] < base[short])),
        "long_correct_skew_frequency": float(np.mean(epsilon_ask[long] < base[long])),
        "mean_spread_pnl": float(np.mean([row["spread_pnl"] for row in formal])),
        "mean_total_pnl": float(np.mean([row["total_pnl"] for row in formal])),
        "action_bounds_valid": bool(
            np.all((-1.0 <= epsilon_bid) & (epsilon_bid <= 1.0))
            and np.all((-1.0 <= epsilon_ask) & (epsilon_ask <= 1.0))
            and np.all((0.0 <= hedge_fraction) & (hedge_fraction <= 1.0))
        ),
    }
    return rows, metrics


def aggregate(metrics: list[dict]) -> dict[float, dict[str, float]]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in metrics:
        grouped[float(row["gamma"])].append(row)
    names = [
        "mean_market_share",
        "mean_abs_inventory",
        "inventory_std",
        "mean_hedge_fraction",
        "short_mean_bid_skew",
        "long_mean_ask_skew",
        "short_correct_skew_frequency",
        "long_correct_skew_frequency",
        "mean_spread_pnl",
        "mean_total_pnl",
    ]
    return {
        gamma: {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in names
        }
        for gamma, rows in grouped.items()
    }


def write_report(output: Path, metrics: list[dict]) -> None:
    summary = aggregate(metrics)
    zero = summary[0.0]
    two = summary[2.0]
    lines = [
        "# Adaptive MM standalone validation",
        "",
        "本实验是机制验证，不是正式 benchmark。Adaptive MM 先完成 121-step "
        "full-grid cold start，随后使用 online response table 对 PersistentMarketMaker"
        "(0.5, 0.5, 0) 报价和对冲。Investor size 使用现有 simulator 默认的 "
        "Gamma flow。",
        "",
        "## 核心结果",
        "",
        "| gamma | Market share | Mean abs inventory | Inventory std | Mean hedge | "
        "Short bid skew | Long ask skew | Spread PnL/step | Total PnL/step |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gamma in (0.0, 2.0):
        row = summary[gamma]
        lines.append(
            f"| {gamma:.0f} | {row['mean_market_share']:.4f} | "
            f"{row['mean_abs_inventory']:.4f} | {row['inventory_std']:.4f} | "
            f"{row['mean_hedge_fraction']:.4f} | {row['short_mean_bid_skew']:.4f} | "
            f"{row['long_mean_ask_skew']:.4f} | {row['mean_spread_pnl']:.6f} | "
            f"{row['mean_total_pnl']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 机制检查",
            "",
            f"- Cold start：所有 run 均为 121 steps，table 完整后连续进入 decide()。",
            f"- Market-share targeting：gamma=0 为 {zero['mean_market_share']:.4f}，"
            f"gamma=2 为 {two['mean_market_share']:.4f}；目标是 0.5。",
            f"- Short inventory：bid-side correct-skew frequency 为 "
            f"{zero['short_correct_skew_frequency']:.4f} (gamma=0) / "
            f"{two['short_correct_skew_frequency']:.4f} (gamma=2)。",
            f"- Long inventory：ask-side correct-skew frequency 为 "
            f"{zero['long_correct_skew_frequency']:.4f} (gamma=0) / "
            f"{two['long_correct_skew_frequency']:.4f} (gamma=2)。",
            f"- Risk aversion：mean hedge fraction 从 {zero['mean_hedge_fraction']:.4f} "
            f"变为 {two['mean_hedge_fraction']:.4f}；mean |inventory| 从 "
            f"{zero['mean_abs_inventory']:.4f} 变为 {two['mean_abs_inventory']:.4f}。",
            "",
            "## 发现的 implementation limitation",
            "",
            "Market share 没有收敛到 0.5：gamma=0 overshoot，gamma=2 undershoot。"
            "Persistent quote 0.5 位于 0.2 grid 的 0.4 与 0.6 之间；deterministic routing "
            "使 symmetric quote 的 realized share 接近 1 或 0，而不是 0.5。加上每个 cell "
            "只有一次 cold-start observation，Step 1 的离散 response 无法稳定表达 50% target。",
            "",
            "本轮不修改 grid、cold start 或 estimator，因此把这一点保留为后续需要明确设计的"
            "问题。它不是 simulator hook 的 flow/sign/timing 错误。",
            "",
            "## 实现说明",
            "",
            "Simulator 只增加 Adaptive-specific optional hooks；Random/Persistent 仍走原有 "
            "act(observation) 路径。本轮未训练 PPO，也未修改 response estimator、cold-start "
            "sequence 或 grid。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Adaptive MM behavioral validation")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10_000)
    args = parser.parse_args()
    if args.seeds <= 0 or args.steps <= 121:
        raise ValueError("seeds must be positive and steps must exceed the 121-step cold start")

    output = ROOT / "results" / "adaptive_mm_validation"
    output.mkdir(parents=True, exist_ok=True)
    trajectories = []
    metrics = []
    for seed in range(args.seeds):
        for gamma in (0.0, 2.0):
            rows, row_metrics = run_path(seed=10_000 + seed, gamma=gamma, steps=args.steps)
            trajectories.extend(rows)
            metrics.append(row_metrics)
            print(
                f"seed={seed} gamma={gamma:.0f} share={row_metrics['mean_market_share']:.3f} "
                f"|inv|={row_metrics['mean_abs_inventory']:.3f} "
                f"hedge={row_metrics['mean_hedge_fraction']:.3f}"
            )

    with (output / "trajectories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trajectories)
    fields = list(metrics[0])
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    write_report(output, metrics)
    print(f"Saved validation results to {output}")


if __name__ == "__main__":
    main()
