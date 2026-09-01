from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avellaneda_stoikov import AvellanedaStoikovMarketMaker
from baselines import PersistentMarketMaker
from config import Config
from environments import TwoDealerMarketEnv
from market import reference_spread


DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_STEPS = 100_352
OUTPUT = ROOT / "results" / "as_benchmark"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_config(seed: int) -> Config:
    return Config(
        seed=seed,
        relative_price=True,
        num_investors=20,
        order_size_mode="unit",
        buy_probability=0.5,
        mu=0.0,
        sigma=0.2,
    )


def summarize_dealer(
    *,
    seed: int,
    agent: str,
    steps: int,
    spread: np.ndarray,
    inventory_pnl: np.ndarray,
    hedge_cost: np.ndarray,
    total: np.ndarray,
    inventory: np.ndarray,
    volume: np.ndarray,
    normalized_notional: np.ndarray,
    extras: dict[str, float],
) -> dict:
    if not np.allclose(spread + inventory_pnl - hedge_cost, total, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"{agent} PnL identity failed")
    captured_volume = float(volume.sum())
    normalized_denominator = float(normalized_notional.sum())
    if captured_volume <= 0.0 or normalized_denominator <= 0.0:
        raise RuntimeError(f"{agent} captured no investor volume")
    return {
        "seed": seed,
        "agent": agent,
        "steps": steps,
        "market_share": captured_volume / (20.0 * steps),
        "captured_volume_per_step": float(volume.mean()),
        "spread_pnl_per_step": float(spread.mean()),
        "spread_pnl_per_captured_unit": float(spread.sum() / captured_volume),
        "normalized_spread_monetization": float(spread.sum() / normalized_denominator),
        "inventory_pnl_per_step": float(inventory_pnl.mean()),
        "hedge_cost_per_step": float(hedge_cost.mean()),
        "total_pnl_per_step": float(total.mean()),
        "mean_abs_inventory": float(np.abs(inventory).mean()),
        "inventory_std": float(inventory.std()),
        "max_abs_inventory": float(np.abs(inventory).max()),
        **extras,
    }


def run_seed(seed: int, steps: int) -> list[dict]:
    cfg = make_config(seed)
    persistent = PersistentMarketMaker(0.0, 0.0, 0.0)
    environment = TwoDealerMarketEnv(
        persistent,
        cfg,
        seed=seed * 100_000 + 13_000,
        reward_mode="total",
    )
    maker = AvellanedaStoikovMarketMaker(
        risk_horizon_steps=26,
        inventory_anchor=20.0,
        neutral_epsilon=0.0,
        hedge_fraction=0.0,
        cfg=cfg,
    )
    environment.reset()
    maker.reset_adaptive_state()

    fields = (
        "spread",
        "inventory_pnl",
        "hedge_cost",
        "total",
        "inventory",
        "volume",
        "normalized_notional",
    )
    series = {
        agent: {field: np.empty(steps, dtype=float) for field in fields}
        for agent in ("A-S", "Persistent")
    }
    inventory_before = np.empty(steps, dtype=float)
    delta_inventory = np.empty(steps, dtype=float)
    raw_bid = np.empty(steps, dtype=float)
    raw_ask = np.empty(steps, dtype=float)

    for step in range(steps):
        inventory_before[step] = environment.inventory[0]
        decision = maker.quote(environment.inventory[0], environment.price)
        action = np.array(
            [decision.epsilon_bid, decision.epsilon_ask, 0.0], dtype=np.float32
        )
        unit_distance = reference_spread(environment.price, 1.0, cfg)
        _, _, _, info = environment.step(action)

        delta_inventory[step] = info["inventory"] - inventory_before[step]
        raw_bid[step] = decision.raw_epsilon_bid
        raw_ask[step] = decision.raw_epsilon_ask
        as_volume = info["gross_investor_volume"] - info["competitor_gross_volume"]
        per_volume = {"A-S": as_volume, "Persistent": info["competitor_gross_volume"]}
        for agent, prefix in (("A-S", ""), ("Persistent", "competitor_")):
            dealer = series[agent]
            dealer["spread"][step] = info[f"{prefix}spread_pnl"]
            dealer["inventory_pnl"][step] = info[f"{prefix}inventory_pnl"]
            dealer["hedge_cost"][step] = info[f"{prefix}hedge_cost"]
            dealer["total"][step] = info[
                "total_pnl" if agent == "A-S" else "competitor_pnl"
            ]
            dealer["inventory"][step] = info[
                "inventory" if agent == "A-S" else "competitor_inventory"
            ]
            dealer["volume"][step] = per_volume[agent]
            dealer["normalized_notional"][step] = per_volume[agent] * unit_distance

    positive = inventory_before > 0.0
    negative = inventory_before < 0.0
    if not positive.any() or not negative.any():
        raise RuntimeError("A-S trajectory did not visit both inventory signs")
    positive_drift = float(delta_inventory[positive].mean())
    negative_drift = float(delta_inventory[negative].mean())
    if positive_drift >= 0.0 or negative_drift <= 0.0:
        raise RuntimeError("A-S inventory flow does not mean-revert in both directions")

    as_extras = {
        "bid_clip_frequency": maker.bid_clip_frequency,
        "ask_clip_frequency": maker.ask_clip_frequency,
        "side_clip_frequency": maker.side_clip_frequency,
        "mean_delta_q_when_q_positive": positive_drift,
        "mean_delta_q_when_q_negative": negative_drift,
        "mean_raw_epsilon_bid": float(raw_bid.mean()),
        "mean_raw_epsilon_ask": float(raw_ask.mean()),
    }
    persistent_extras = {key: 0.0 for key in as_extras}
    return [
        summarize_dealer(
            seed=seed,
            agent=agent,
            steps=steps,
            extras=as_extras if agent == "A-S" else persistent_extras,
            **series[agent],
        )
        for agent in ("A-S", "Persistent")
    ]


def mean_std(rows: list[dict], agent: str, metric: str) -> tuple[float, float]:
    values = np.asarray(
        [float(row[metric]) for row in rows if row["agent"] == agent], dtype=float
    )
    return float(values.mean()), float(values.std())


def formatted(value: tuple[float, float], digits: int = 4) -> str:
    return f"{value[0]:.{digits}f} ± {value[1]:.{digits}f}"


def write_report(rows: list[dict], output: Path) -> None:
    cfg = make_config(DEFAULT_SEEDS[0])
    maker = AvellanedaStoikovMarketMaker(cfg=cfg)
    gamma, k = maker.canonical_parameters(cfg.P0)
    agents = ("A-S", "Persistent")
    metrics = (
        ("market_share", "Market share", 4),
        ("spread_pnl_per_step", "Spread PnL/step", 4),
        ("spread_pnl_per_captured_unit", "Spread/unit", 4),
        ("normalized_spread_monetization", "Normalized spread monetization", 4),
        ("inventory_pnl_per_step", "Inventory PnL/step", 4),
        ("hedge_cost_per_step", "Hedge cost/step", 4),
        ("total_pnl_per_step", "Total PnL/step", 4),
        ("mean_abs_inventory", "E|q|", 3),
        ("inventory_std", "std(q)", 3),
    )
    clip = mean_std(rows, "A-S", "side_clip_frequency")
    positive_drift = mean_std(rows, "A-S", "mean_delta_q_when_q_positive")
    negative_drift = mean_std(rows, "A-S", "mean_delta_q_when_q_negative")
    clip_assessment = (
        "低于 10% 的预设理想阈值，保留 inventory_anchor=20。"
        if clip[0] < 0.10
        else "达到或超过 10%；进入 PPO benchmark 前需要复核 anchor。"
    )
    lines = [
        "# Normalized stationary A-S benchmark",
        "",
        "## 定义",
        "",
        "Canonical A-S 保留 `r = S - q gamma V` 与 `Delta = gamma V + "
        "2/gamma log(1 + gamma/k)`，其中 `V = P^2 sigma^2 dt H`。正式 benchmark "
        "固定 normalized coordinates `rho=gamma D`、`kappa=kD`，而不是从当前 "
        "winner-take-all routing 拟合不存在的 smooth execution elasticity。",
        "",
        f"- risk_horizon_steps = {maker.risk_horizon_steps}（一个 6.5 小时交易日）",
        f"- neutral_epsilon = {maker.neutral_epsilon:g}",
        f"- inventory_anchor = {maker.inventory_anchor:g}",
        f"- hedge_fraction = {maker.hedge_fraction:g}",
        f"- rho = {maker.rho:.12g}",
        f"- kappa = {maker.kappa:.12g}",
        f"- P0 下等价 gamma = {gamma:.12g}，k = {k:.12g}",
        "",
        "这是 stationary scale-normalized A-S implementation：normalized control parameters "
        "固定，其 dollar-coordinate equivalents 随 prevailing unit reference spread 缩放。",
        "",
        "## Compatibility choices",
        "",
        "- fixed receding one-day risk horizon，不使用训练 horizon；",
        "- GBM local Brownian variance approximation；",
        "- zero-inventory quote 对齐 simulator unit-flow reference quote；",
        "- inventory_anchor 来自每步 20 units 的市场流量，而非 PPO/PnL calibration；",
        "- native winner-take-all routing、native price/RNG/PnL accounting；",
        "- external hedge 固定为零；",
        "- executable epsilon 按 simulator action domain 裁剪，并显式记录 raw quote 与 clip。",
        "",
        "## Standalone setup",
        "",
        f"{len({int(row['seed']) for row in rows})} seeds × {int(rows[0]['steps']):,} steps，"
        "unit flow，sigma=0.2。A-S 与 Persistent(epsilon_bid=epsilon_ask=0, hedge=0) "
        "共享同一个 TwoDealerMarketEnv、investor orders 与 price path。",
        "",
        "## 三 seed 结果（population std）",
        "",
        "| Metric | A-S | Persistent |",
        "|---|---:|---:|",
    ]
    for key, label, digits in metrics:
        lines.append(
            f"| {label} | {formatted(mean_std(rows, agents[0], key), digits)} | "
            f"{formatted(mean_std(rows, agents[1], key), digits)} |"
        )
    lines.extend(
        [
            "",
            "## Inventory dynamics and clipping",
            "",
            f"- E[Delta q | q>0] = {formatted(positive_drift)} < 0。",
            f"- E[Delta q | q<0] = {formatted(negative_drift)} > 0。",
            f"- Bid clip frequency = {formatted(mean_std(rows, 'A-S', 'bid_clip_frequency'))}。",
            f"- Ask clip frequency = {formatted(mean_std(rows, 'A-S', 'ask_clip_frequency'))}。",
            f"- Pooled side clip frequency = {formatted(clip)}；{clip_assessment}",
            "",
            "两项 conditional drift 的符号直接确认 analytical skew 在 native execution "
            "中会清算库存。所有逐步数据均满足 Spread + Inventory - HedgeCost = Total，"
            "所有统计量 finite。Standalone gate 通过后，上述 A-S 参数冻结用于 PPO 对比。",
            "",
        ]
    )
    (output / "report_zh.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")

    rows: list[dict] = []
    for seed in args.seeds:
        seed_rows = run_seed(seed, args.steps)
        rows.extend(seed_rows)
        as_row = seed_rows[0]
        print(
            f"seed={seed} A-S share={as_row['market_share']:.4f} "
            f"E|q|={as_row['mean_abs_inventory']:.3f} "
            f"side_clip={as_row['side_clip_frequency']:.4f}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "metrics.csv", rows)
    write_report(rows, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
