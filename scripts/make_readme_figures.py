from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"

STABILIZATION_STAGES = (
    ("Shared actor/critic", ROOT / "results" / "long_run_random" / "metrics_by_seed.csv"),
    (
        "Separate actor/critic",
        ROOT / "results" / "separate_actor_critic" / "metrics_by_seed.csv",
    ),
    ("+ Relative price", ROOT / "results" / "relative_price" / "metrics_by_seed.csv"),
    (
        "+ State-dependent std",
        ROOT / "results" / "state_dependent_std" / "metrics_by_seed.csv",
    ),
)
INVENTORY_SIGMA_PATH = (
    ROOT / "results" / "inventory_variance_analysis" / "inventory_sigma_bins.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def final_five_seed_mae(path: Path) -> tuple[float, float]:
    rows = read_csv(path)
    seeds = {int(row["seed"]) for row in rows}
    if seeds != set(range(5)):
        raise ValueError(f"expected seeds 0-4 in {path}, found {sorted(seeds)}")
    if any(int(row["training_steps"]) != 204_800 for row in rows):
        raise ValueError(f"expected 204,800-step results in {path}")

    per_seed_mae = [
        (
            abs(float(row["deterministic_epsilon_bid_mean"]))
            + abs(float(row["deterministic_epsilon_ask_mean"]))
        )
        / 2.0
        for row in rows
    ]
    if not all(math.isfinite(value) for value in per_seed_mae):
        raise ValueError(f"non-finite MAE in {path}")
    return statistics.mean(per_seed_mae), statistics.pstdev(per_seed_mae)


def make_stabilization_figure(output: Path) -> list[float]:
    labels = []
    means = []
    dispersions = []
    for label, path in STABILIZATION_STAGES:
        mean, dispersion = final_five_seed_mae(path)
        labels.append(label)
        means.append(mean)
        dispersions.append(dispersion)

    figure, axis = plt.subplots(figsize=(10.2, 5.7))
    colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#59A14F"]
    bars = axis.bar(
        labels,
        means,
        yerr=dispersions,
        capsize=5,
        color=colors,
        edgecolor="none",
        error_kw={"ecolor": "#3F3F3F", "elinewidth": 1.2, "capthick": 1.2},
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=6)
    axis.set_title("PPO stabilization journey", fontsize=16, fontweight="semibold", pad=14)
    axis.set_xlabel("Development stage")
    axis.set_ylabel("Final five-seed two-side MAE")
    axis.set_ylim(0.0, max(mean + std for mean, std in zip(means, dispersions)) * 1.18)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelrotation=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return means


def aggregate_inventory_sigma() -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(INVENTORY_SIGMA_PATH):
        grouped[row["inventory_bin"]].append(row)

    expected_bins = {f"Q{index}" for index in range(1, 6)}
    if set(grouped) != expected_bins:
        raise ValueError(f"expected bins Q1-Q5, found {sorted(grouped)}")

    aggregated = []
    for inventory_bin in sorted(grouped, key=lambda value: int(value[1:])):
        rows = grouped[inventory_bin]
        if {int(row["seed"]) for row in rows} != set(range(5)):
            raise ValueError(f"expected seeds 0-4 in {inventory_bin}")
        aggregated.append(
            {
                "inventory_bin": inventory_bin,
                "inventory_mean": statistics.mean(
                    float(row["inventory_mean"]) for row in rows
                ),
                "sigma_bid_mean": statistics.mean(
                    float(row["sigma_bid_mean"]) for row in rows
                ),
                "sigma_ask_mean": statistics.mean(
                    float(row["sigma_ask_mean"]) for row in rows
                ),
            }
        )
    return aggregated


def make_inventory_figure(output: Path) -> list[dict[str, float | str]]:
    rows = aggregate_inventory_sigma()
    inventory = [float(row["inventory_mean"]) for row in rows]
    sigma_bid = [float(row["sigma_bid_mean"]) for row in rows]
    sigma_ask = [float(row["sigma_ask_mean"]) for row in rows]

    figure, axis = plt.subplots(figsize=(9.4, 5.7))
    axis.plot(
        inventory,
        sigma_bid,
        color="#4C78A8",
        marker="o",
        markersize=7,
        linewidth=2.3,
        label="Bid σ(s)",
    )
    axis.plot(
        inventory,
        sigma_ask,
        color="#F28E2B",
        marker="s",
        markersize=7,
        linewidth=2.3,
        label="Ask σ(s)",
    )
    axis.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_title(
        "Inventory-conditioned exploration", fontsize=16, fontweight="semibold", pad=14
    )
    axis.set_xlabel("Mean inventory before action")
    axis.set_ylabel("Mean latent standard deviation")
    axis.grid(color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return rows


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stabilization_path = FIGURES_DIR / "ppo_stabilization_journey.png"
    inventory_path = FIGURES_DIR / "inventory_conditioned_exploration.png"

    stage_means = make_stabilization_figure(stabilization_path)
    inventory_rows = make_inventory_figure(inventory_path)

    print(f"Generated {stabilization_path.relative_to(ROOT)}")
    print("  MAE: " + ", ".join(f"{value:.4f}" for value in stage_means))
    print(f"Generated {inventory_path.relative_to(ROOT)}")
    print(
        "  Inventory range: "
        f"{float(inventory_rows[0]['inventory_mean']):.3f} to "
        f"{float(inventory_rows[-1]['inventory_mean']):.3f}"
    )


if __name__ == "__main__":
    main()
