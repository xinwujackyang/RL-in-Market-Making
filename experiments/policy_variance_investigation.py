from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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


LEGENDRE_NODES, LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(240)


def expected_spread_factor(latent_mean: float, latent_std: float, scenario: str) -> float:
    if scenario == "persistent_spread_only":
        upper = min(8.0, (np.arctanh(0.5) - latent_mean) / latent_std)
        if upper <= -8.0:
            return 0.0
        z = 0.5 * (upper + 8.0) * LEGENDRE_NODES + 0.5 * (upper - 8.0)
        scale = 0.5 * (upper + 8.0)
        epsilon = np.tanh(latent_mean + latent_std * z)
        payoff = 1.0 + epsilon
    elif scenario == "random":
        z = 8.0 * LEGENDRE_NODES
        scale = 8.0
        epsilon = np.tanh(latent_mean + latent_std * z)
        payoff = 0.5 * (1.0 - epsilon**2)
    else:
        raise ValueError(scenario)
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return float(scale * np.sum(LEGENDRE_WEIGHTS * density * payoff))


def stochastic_optimum(latent_std: float, scenario: str) -> dict:
    mean_grid = np.linspace(-0.8, 1.0, 901)
    values = np.asarray([expected_spread_factor(mean, latent_std, scenario) for mean in mean_grid])
    best = int(values.argmax())
    return {
        "scenario": scenario,
        "fixed_std": latent_std,
        "optimal_latent_mean": float(mean_grid[best]),
        "optimal_deterministic_epsilon": float(np.tanh(mean_grid[best])),
        "optimal_expected_spread_factor": float(values[best]),
    }


def slope_metrics(data: dict[str, np.ndarray]) -> dict:
    inventory = data["inventory_before_action"]
    bid = data["epsilon_bid"]
    ask = data["epsilon_ask"]
    return {
        "inventory_bid_slope": float(np.polyfit(inventory, bid, 1)[0]),
        "inventory_ask_slope": float(np.polyfit(inventory, ask, 1)[0]),
        "inventory_skew_slope": float(np.polyfit(inventory, bid - ask, 1)[0]),
    }


def binned_curve(data: dict[str, np.ndarray], bins: int = 10) -> list[dict]:
    inventory = data["inventory_before_action"]
    edges = np.unique(np.quantile(inventory, np.linspace(0.0, 1.0, bins + 1)))
    membership = np.digitize(inventory, edges[1:-1])
    rows = []
    for index in range(len(edges) - 1):
        chosen = membership == index
        if chosen.sum() < 10:
            continue
        bid = data["epsilon_bid"][chosen]
        ask = data["epsilon_ask"][chosen]
        rows.append(
            {
                "inventory": float(inventory[chosen].mean()),
                "epsilon_bid": float(bid.mean()),
                "epsilon_ask": float(ask.mean()),
                "epsilon_skew": float((bid - ask).mean()),
                "count": int(chosen.sum()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows])
    return float(values.mean()), float(values.std())


def make_env_factory(scenario: str, cfg: Config, seed: int):
    def env_factory(episode: int):
        if scenario == "random":
            competitor = RandomMarketMaker(
                hedge_fraction_max=0.3, seed=seed * 100_000 + 20_000 + episode
            )
            reward_mode = "total"
        else:
            competitor = PersistentMarketMaker(0.5, 0.5, 0.0)
            reward_mode = "spread"
        return TwoDealerMarketEnv(
            competitor,
            cfg,
            seed=seed * 100_000 + 10_000 + episode,
            reward_mode=reward_mode,
        )

    return env_factory


def plot_training_diagnostics(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row_index, scenario in enumerate(("random", "persistent_spread_only")):
        selected = [row for row in rows if row["scenario"] == scenario and row["std_setting"] == "learned"]
        steps = sorted({int(row["step"]) for row in selected})
        for action in ("bid", "ask", "hedge"):
            means = [np.mean([float(row[f"latent_std_{action}"]) for row in selected if int(row["step"]) == step]) for step in steps]
            axes[row_index, 0].plot(steps, means, label=action)
        entropy = [np.mean([float(row["entropy_proxy"]) for row in selected if int(row["step"]) == step]) for step in steps]
        grad_std = [np.mean([float(row["mean_abs_grad_log_std"]) for row in selected if int(row["step"]) == step]) for step in steps]
        grad_mean = [np.mean([float(row["mean_abs_grad_actor_mean"]) for row in selected if int(row["step"]) == step]) for step in steps]
        axes[row_index, 1].plot(steps, entropy, color="tab:purple")
        axes[row_index, 2].semilogy(steps, grad_std, label="|grad log_std|")
        axes[row_index, 2].semilogy(steps, grad_mean, label="|grad actor mean head|")
        axes[row_index, 2].axhline(3e-3, color="black", linestyle=":", label="|entropy-only grad|=0.003")
        axes[row_index, 0].set(title=f"{scenario}: latent std", xlabel="training step")
        axes[row_index, 1].set(title=f"{scenario}: entropy proxy", xlabel="training step")
        axes[row_index, 2].set(title=f"{scenario}: pre-clip gradients", xlabel="training step")
        axes[row_index, 0].legend()
        axes[row_index, 2].legend()
    figure.tight_layout()
    figure.savefig(output / "variance_gradient_trajectories.png", dpi=160)
    plt.close(figure)


def plot_mean_vs_sampled(curves: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row_index, scenario in enumerate(("random", "persistent_spread_only")):
        for mode, style in (("deterministic", "-"), ("stochastic", "--")):
            chosen = sorted(
                [row for row in curves if row["scenario"] == scenario and row["std_setting"] == "learned" and row["mode"] == mode],
                key=lambda row: row["inventory"],
            )
            inventory = [row["inventory"] for row in chosen]
            for column, key in enumerate(("epsilon_bid", "epsilon_ask", "epsilon_skew")):
                axes[row_index, column].plot(inventory, [row[key] for row in chosen], style, marker="o", label=mode)
                axes[row_index, column].axhline(0.0, color="black", linewidth=0.7)
                axes[row_index, column].set(title=f"{scenario}: {key}", xlabel="inventory")
                axes[row_index, column].legend()
    figure.tight_layout()
    figure.savefig(output / "deterministic_vs_stochastic_inventory_skew.png", dpi=160)
    plt.close(figure)


def plot_per_seed_slopes(metrics: list[dict], output: Path) -> None:
    selected = [
        row for row in metrics
        if row["scenario"] == "random" and row["std_setting"] == "learned"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    seeds = sorted({int(row["seed"]) for row in selected})
    for axis, key, expected in zip(
        axes,
        ("inventory_bid_slope", "inventory_ask_slope", "inventory_skew_slope"),
        ("positive", "negative", "positive"),
    ):
        for mode, marker in (("deterministic", "o"), ("stochastic", "s")):
            values = [
                float(next(row for row in selected if int(row["seed"]) == seed and row["mode"] == mode)[key])
                for seed in seeds
            ]
            axis.plot(seeds, values, marker=marker, label=mode)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=f"{key}\nexpected {expected}", xlabel="seed", ylabel="slope")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "per_seed_inventory_slopes.png", dpi=160)
    plt.close(figure)


def plot_fixed_std(metrics: list[dict], optima: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    std_values = [0.1, 0.2, 0.5, 1.0]
    for row_index, scenario in enumerate(("random", "persistent_spread_only")):
        deterministic = [row for row in metrics if row["scenario"] == scenario and row["mode"] == "deterministic" and row["std_setting"] != "learned"]
        stochastic = [row for row in metrics if row["scenario"] == scenario and row["mode"] == "stochastic" and row["std_setting"] != "learned"]
        analytical = {float(row["fixed_std"]): row for row in optima if row["scenario"] == scenario}
        for key, label in (("mean_epsilon_bid", "bid"), ("mean_epsilon_ask", "ask")):
            means, errors = [], []
            for std in std_values:
                group = [row for row in deterministic if float(row["fixed_std"]) == std]
                mean, error = group_mean_std(group, key)
                means.append(mean); errors.append(error)
            axes[row_index, 0].errorbar(std_values, means, yerr=errors, marker="o", label=label)
        axes[row_index, 0].plot(std_values, [analytical[std]["optimal_deterministic_epsilon"] for std in std_values], color="black", linestyle="--", label="stochastic optimum")
        for key, label in (("epsilon_bid_std", "sampled bid std"), ("epsilon_ask_std", "sampled ask std")):
            means = [group_mean_std([row for row in stochastic if float(row["fixed_std"]) == std], key)[0] for std in std_values]
            axes[row_index, 1].plot(std_values, means, marker="o", label=label)
        spread = [group_mean_std([row for row in stochastic if float(row["fixed_std"]) == std], "mean_spread_pnl")[0] for std in std_values]
        share = [group_mean_std([row for row in stochastic if float(row["fixed_std"]) == std], "mean_market_share")[0] for std in std_values]
        axes[row_index, 2].plot(std_values, spread, marker="o", label="spread PnL/step")
        axes[row_index, 2].plot(std_values, share, marker="o", label="market share")
        for column in range(3):
            axes[row_index, column].set_xscale("log")
            axes[row_index, column].set_xlabel("fixed latent sigma")
            axes[row_index, column].set_title(scenario)
            axes[row_index, column].legend()
        axes[row_index, 0].set_ylabel("deterministic mean epsilon")
    figure.tight_layout()
    figure.savefig(output / "fixed_sigma_best_response.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy variance and inventory-skew investigation, phases 1-5")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    output = ROOT / "results" / "policy_variance"
    output.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    curve_data: dict[tuple[str, str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    settings = [("learned", None), ("1.0", 1.0), ("0.5", 0.5), ("0.2", 0.2), ("0.1", 0.1)]

    for scenario in ("random", "persistent_spread_only"):
        for std_setting, fixed_std in settings:
            for seed in range(args.seeds):
                cfg = Config(
                    seed=seed,
                    rollout=args.rollouts,
                    horizon=args.horizon,
                    eval_episodes=args.eval_episodes,
                    hidden_layers=3,
                    sigma=0.0 if scenario == "persistent_spread_only" else Config.sigma,
                    fixed_policy_std=fixed_std,
                )
                env_factory = make_env_factory(scenario, cfg, seed)
                agent = PPOAgent(env_factory(0), cfg)
                history = agent.train()
                for row in history.diagnostics:
                    diagnostic_rows.append({"scenario": scenario, "std_setting": std_setting, "fixed_std": fixed_std if fixed_std is not None else "", "seed": seed, **row})

                for mode, deterministic in (("deterministic", True), ("stochastic", False)):
                    metrics, data = evaluate_policy(
                        agent,
                        env_factory,
                        args.eval_episodes,
                        cfg.steps_per_day * cfg.eval_days,
                        deterministic=deterministic,
                    )
                    metrics.update(slope_metrics(data))
                    metrics.update(
                        scenario=scenario,
                        std_setting=std_setting,
                        fixed_std=fixed_std if fixed_std is not None else "",
                        seed=seed,
                        mode=mode,
                        final_rollout_reward=history.rewards[-1],
                    )
                    metrics_rows.append(metrics)
                    curve_data[(scenario, std_setting, mode)].append(data)
                print(f"{scenario:24s} sigma={std_setting:7s} seed={seed} complete")

    curve_rows = []
    for (scenario, std_setting, mode), datasets in curve_data.items():
        merged = {key: np.concatenate([data[key] for data in datasets]) for key in datasets[0]}
        for row in binned_curve(merged):
            curve_rows.append({"scenario": scenario, "std_setting": std_setting, "mode": mode, **row})

    optima_rows = [stochastic_optimum(std, scenario) for scenario in ("random", "persistent_spread_only") for std in (1.0, 0.5, 0.2, 0.1)]
    write_csv(output / "metrics.csv", metrics_rows)
    write_csv(output / "training_diagnostics.csv", diagnostic_rows)
    write_csv(output / "inventory_curves.csv", curve_rows)
    write_csv(output / "analytical_optima.csv", optima_rows)
    plot_training_diagnostics(diagnostic_rows, output)
    plot_mean_vs_sampled(curve_rows, output)
    plot_per_seed_slopes(metrics_rows, output)
    plot_fixed_std(metrics_rows, optima_rows, output)
    print(f"Saved Phase 1-5 investigation outputs to {output}")


if __name__ == "__main__":
    main()
