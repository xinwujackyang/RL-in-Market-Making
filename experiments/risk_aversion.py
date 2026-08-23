from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import PersistentMarketMaker
from config import Config
from environments import TwoDealerMarketEnv
from evaluation import evaluate_policy
from ppo import PPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired current-step inventory-PnL-squared risk study")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    output = ROOT / "results" / "risk_aversion"
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in range(args.seeds):
        cfg = Config(seed=seed, rollout=args.rollouts, horizon=args.horizon, eval_episodes=args.eval_episodes, hidden_layers=3)

        def env_factory(episode: int):
            return TwoDealerMarketEnv(
                PersistentMarketMaker(0.5, 0.5, 0.0), cfg, seed=seed * 10_000 + 3_000 + episode
            )

        for penalized in (False, True):
            agent = PPOAgent(env_factory(0), cfg)
            agent.train(risk_penalty=penalized)
            metrics, _ = evaluate_policy(
                agent, env_factory, cfg.eval_episodes, cfg.steps_per_day * cfg.eval_days, deterministic=False
            )
            metrics.update(seed=seed, penalty="current_inv_pnl_squared" if penalized else "none", alpha=cfg.risk_penalty_alpha if penalized else 0.0)
            rows.append(metrics)
            print(
                f"seed={seed} penalty={metrics['penalty']}: total={metrics['mean_total_pnl']:.2f}, "
                f"inventory_std={metrics['inventory_std']:.2f}, invpnl_std={metrics['inventory_pnl_std']:.2f}"
            )

    fields = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics_to_plot = ["mean_total_pnl", "pnl_std", "inventory_std", "inventory_pnl_std", "mean_hedge_fraction", "mean_market_share"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    penalties = ["none", "current_inv_pnl_squared"]
    for axis, metric in zip(axes.flat, metrics_to_plot):
        values = [[float(row[metric]) for row in rows if row["penalty"] == penalty] for penalty in penalties]
        axis.boxplot(values, tick_labels=["none", "InvPnL²"])
        for seed in range(args.seeds):
            axis.plot([1, 2], [values[0][seed], values[1][seed]], color="gray", alpha=0.5)
        axis.set_title(metric.replace("_", " "))
    figure.suptitle(f"Risk-return comparison, alpha={Config.risk_penalty_alpha}")
    figure.tight_layout()
    figure.savefig(output / "risk_return_comparison.png", dpi=160)
    plt.close(figure)
    print(f"Saved {output / 'metrics.csv'} and {output / 'risk_return_comparison.png'}")


if __name__ == "__main__":
    main()
