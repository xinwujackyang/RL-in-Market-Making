from __future__ import annotations

from collections import defaultdict

import numpy as np


def evaluate_policy(
    agent, env_factory, episodes: int, steps: int, deterministic: bool = True
) -> tuple[dict, dict[str, np.ndarray]]:
    episode_totals = []
    episode_competitor_totals = []
    series: dict[str, list[float]] = defaultdict(list)
    for episode in range(episodes):
        env = env_factory(episode)
        observation = env.reset()
        total = competitor_total = 0.0
        for _ in range(steps):
            series["inventory_before_action"].append(float(observation[0]))
            action = (
                agent.deterministic_action(observation)
                if deterministic
                else agent.sample_action(observation)
            )
            series["policy_entropy"].append(agent.policy_entropy(observation))
            observation, reward, _, info = env.step(action)
            # Report economic total PnL even when a diagnostic environment is
            # intentionally trained on a spread-only reward.
            total += info["total_pnl"]
            competitor_total += info.get("competitor_pnl", 0.0)
            series["inventory"].append(info["inventory"])
            series["spread_pnl"].append(info["spread_pnl"])
            series["inventory_pnl"].append(info["inventory_pnl"])
            series["hedge_cost"].append(info["hedge_cost"])
            series["epsilon_bid"].append(float(action[0]))
            series["epsilon_ask"].append(float(action[1]))
            series["hedge_fraction"].append(float(action[2]))
            if "market_share" in info:
                series["market_share"].append(info["market_share"])
            for key in (
                "n_buy",
                "n_sell",
                "n_rl_won",
                "n_competitor_won",
                "gross_investor_volume",
                "net_investor_flow",
            ):
                if key in info:
                    series[key].append(info[key])
        episode_totals.append(total)
        if hasattr(env, "competitor"):
            episode_competitor_totals.append(competitor_total)

    data = {key: np.asarray(values) for key, values in series.items()}
    metrics = {
        "deterministic_evaluation": deterministic,
        "mean_total_pnl": float(np.mean(episode_totals)),
        "mean_total_pnl_per_step": float(np.mean(episode_totals) / steps),
        "pnl_std": float(np.std(episode_totals)),
        "mean_spread_pnl": float(data["spread_pnl"].mean()),
        "spread_pnl_std": float(data["spread_pnl"].std()),
        "mean_inventory_pnl": float(data["inventory_pnl"].mean()),
        "inventory_pnl_std": float(data["inventory_pnl"].std()),
        "mean_hedge_cost": float(data["hedge_cost"].mean()),
        "mean_inventory": float(data["inventory"].mean()),
        "inventory_std": float(data["inventory"].std()),
        "max_abs_inventory": float(np.abs(data["inventory"]).max()),
        "mean_epsilon_bid": float(data["epsilon_bid"].mean()),
        "epsilon_bid_std": float(data["epsilon_bid"].std()),
        "mean_epsilon_ask": float(data["epsilon_ask"].mean()),
        "epsilon_ask_std": float(data["epsilon_ask"].std()),
        "mean_hedge_fraction": float(data["hedge_fraction"].mean()),
        "hedge_fraction_std": float(data["hedge_fraction"].std()),
        "epsilon_near_bound_frequency": float(
            np.mean(
                np.concatenate(
                    [np.abs(data["epsilon_bid"]) > 0.99, np.abs(data["epsilon_ask"]) > 0.99]
                )
            )
        ),
        "hedge_near_bound_frequency": float(
            np.mean((data["hedge_fraction"] < 0.01) | (data["hedge_fraction"] > 0.99))
        ),
        "mean_policy_entropy_proxy": float(data["policy_entropy"].mean()),
    }
    if hasattr(agent, "network"):
        latent_std = agent.network.log_std.detach().exp().cpu().numpy()
        metrics.update(
            latent_std_bid=float(latent_std[0]),
            latent_std_ask=float(latent_std[1]),
            latent_std_hedge=float(latent_std[2]),
        )
    if episode_competitor_totals:
        metrics["mean_competitor_pnl"] = float(np.mean(episode_competitor_totals))
        metrics["competitor_pnl_std"] = float(np.std(episode_competitor_totals))
        metrics["mean_market_share"] = float(data["market_share"].mean())
        for key in (
            "n_buy",
            "n_sell",
            "n_rl_won",
            "n_competitor_won",
            "gross_investor_volume",
            "net_investor_flow",
            "market_share",
        ):
            metrics[f"{key}_mean"] = float(data[key].mean())
            metrics[f"{key}_std"] = float(data[key].std())
    return metrics, data


def print_metrics(metrics: dict) -> None:
    print("\nEvaluation")
    for name, value in metrics.items():
        label = name.replace("_", " ")
        if isinstance(value, bool):
            print(f"  {label:26s} {str(value):>12s}")
        else:
            print(f"  {label:26s} {value:12.4f}")


def evaluate_baseline(policy, env_factory, episodes: int, steps: int) -> dict:
    totals = []
    for episode in range(episodes):
        env = env_factory(episode)
        observation = env.reset()
        total = 0.0
        for _ in range(steps):
            observation, reward, _, _ = env.step(policy.act(observation))
            total += reward
        totals.append(total)
    return {"mean_total_pnl": float(np.mean(totals)), "pnl_std": float(np.std(totals))}


def save_plots(history, data: dict[str, np.ndarray], output_path: str, competitor: bool = False) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(history.steps, history.rewards)
    axes[0, 0].set(title="Training reward", xlabel="Steps")
    axes[0, 1].hist(data["inventory"], bins=40)
    axes[0, 1].set(title="Inventory distribution")
    axes[0, 2].hist(data["epsilon_bid"], bins=40, alpha=0.65, label="bid")
    axes[0, 2].hist(data["epsilon_ask"], bins=40, alpha=0.65, label="ask")
    axes[0, 2].set(title="Epsilon distributions")
    axes[0, 2].legend()
    labels = ["spread", "inventory", "hedge cost"]
    values = [data["spread_pnl"].mean(), data["inventory_pnl"].mean(), data["hedge_cost"].mean()]
    axes[1, 0].bar(labels, values)
    axes[1, 0].set(title="Mean per-step RL PnL decomposition")
    evaluations = history.evaluation
    if evaluations:
        eval_steps = [item["step"] for item in evaluations]
        axes[1, 1].plot(eval_steps, [item["mean_total_pnl"] for item in evaluations], label="RL")
        if competitor and "mean_competitor_pnl" in evaluations[0]:
            axes[1, 1].plot(eval_steps, [item["mean_competitor_pnl"] for item in evaluations], label="competitor")
        axes[1, 1].legend()
    axes[1, 1].set(title="Evaluation PnL", xlabel="Steps")
    sample = np.arange(len(data["epsilon_bid"]))
    axes[1, 2].plot(sample, data["epsilon_bid"], alpha=0.75, label="bid")
    axes[1, 2].plot(sample, data["epsilon_ask"], alpha=0.75, label="ask")
    axes[1, 2].set(title="Evaluation epsilon path", xlabel="Step")
    axes[1, 2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
