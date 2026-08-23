from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import PersistentMarketMaker, RandomMarketMaker
from config import Config
from environments import TwoDealerMarketEnv
from evaluation import evaluate_policy
from ppo import PPOAgent


def train_seed(scenario: str, seed: int, rollouts: int, horizon: int, eval_episodes: int):
    spread_only = scenario == "persistent_spread_only"
    cfg = Config(
        seed=seed,
        rollout=rollouts,
        horizon=horizon,
        eval_episodes=eval_episodes,
        hidden_layers=3,
        sigma=0.0 if spread_only else Config.sigma,
    )

    def env_factory(episode: int):
        if scenario == "random":
            competitor = RandomMarketMaker(hedge_fraction_max=0.3, seed=seed * 10_000 + 2_000 + episode)
        else:
            competitor = PersistentMarketMaker(0.5, 0.5, 0.0)
        return TwoDealerMarketEnv(
            competitor,
            cfg,
            seed=seed * 10_000 + 1_000 + episode,
            reward_mode="spread" if spread_only else "total",
        )

    agent = PPOAgent(env_factory(0), cfg)
    steps = cfg.steps_per_day * cfg.eval_days
    evaluator = lambda current: evaluate_policy(current, env_factory, cfg.eval_episodes, steps)[0]
    history = agent.train(evaluator=evaluator)
    deterministic_metrics, _ = evaluate_policy(agent, env_factory, cfg.eval_episodes, steps)
    metrics, data = evaluate_policy(agent, env_factory, cfg.eval_episodes, steps, deterministic=False)
    metrics.update(
        deterministic_mean_epsilon_bid=deterministic_metrics["mean_epsilon_bid"],
        deterministic_mean_epsilon_ask=deterministic_metrics["mean_epsilon_ask"],
        deterministic_mean_hedge_fraction=deterministic_metrics["mean_hedge_fraction"],
        deterministic_mean_total_pnl=deterministic_metrics["mean_total_pnl"],
    )
    metrics.update(scenario=scenario, seed=seed, analytical_epsilon=0.0 if scenario == "random" else 0.5)
    print(
        f"{scenario} seed={seed}: sampled eps=({metrics['mean_epsilon_bid']:.3f},"
        f" {metrics['mean_epsilon_ask']:.3f}), deterministic eps="
        f"({metrics['deterministic_mean_epsilon_bid']:.3f},"
        f" {metrics['deterministic_mean_epsilon_ask']:.3f}), share={metrics['mean_market_share']:.3f}"
    )
    return metrics, data, history


def binned_skew(data: dict[str, np.ndarray], bins: int = 10):
    inventory = data["inventory_before_action"]
    edges = np.unique(np.quantile(inventory, np.linspace(0, 1, bins + 1)))
    indices = np.digitize(inventory, edges[1:-1])
    centers, bid, ask = [], [], []
    for index in range(len(edges) - 1):
        selected = indices == index
        if selected.sum() < 5:
            continue
        centers.append(inventory[selected].mean())
        bid.append(data["epsilon_bid"][selected].mean())
        ask.append(data["epsilon_ask"][selected].mean())
    return np.asarray(centers), np.asarray(bid), np.asarray(ask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-seed analytical best-response replication")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    output = ROOT / "results" / "replication"
    output.mkdir(parents=True, exist_ok=True)

    rows, all_data, histories = [], {}, {}
    for scenario in ("random", "persistent", "persistent_spread_only"):
        scenario_data = []
        scenario_histories = []
        for seed in range(args.seeds):
            metrics, data, history = train_seed(scenario, seed, args.rollouts, args.horizon, args.eval_episodes)
            rows.append(metrics)
            scenario_data.append(data)
            scenario_histories.append(history)
        all_data[scenario] = {key: np.concatenate([item[key] for item in scenario_data]) for key in scenario_data[0]}
        histories[scenario] = scenario_histories

    fields = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for column, scenario in enumerate(("random", "persistent", "persistent_spread_only")):
        data = all_data[scenario]
        benchmark = 0.0 if scenario == "random" else 0.5
        axes[0, column].hist(data["epsilon_bid"], bins=45, density=True, alpha=0.6, label="bid")
        axes[0, column].hist(data["epsilon_ask"], bins=45, density=True, alpha=0.6, label="ask")
        axes[0, column].axvline(benchmark, color="black", linestyle="--", label="analytical")
        axes[0, column].set(title=scenario.replace("_", " ") + " epsilon", xlabel="epsilon")
        axes[0, column].legend()
        centers, bid, ask = binned_skew(data)
        axes[1, column].plot(centers, bid, marker="o", label="bid")
        axes[1, column].plot(centers, ask, marker="o", label="ask")
        axes[1, column].plot(centers, bid - ask, marker=".", linestyle="--", label="bid-ask skew")
        axes[1, column].axhline(0, color="black", linewidth=0.7)
        axes[1, column].set(title="Inventory-conditional quotes", xlabel="dealer inventory")
        axes[1, column].legend()
    figure.tight_layout()
    figure.savefig(output / "best_response_and_skew.png", dpi=160)
    plt.close(figure)
    print(f"Saved {output / 'metrics.csv'} and {output / 'best_response_and_skew.png'}")


if __name__ == "__main__":
    main()
