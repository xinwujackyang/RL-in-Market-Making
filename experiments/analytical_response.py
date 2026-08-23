from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGENDRE_NODES, LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(240)


def expected_spread_factor(latent_mean: float, latent_std: float, competitor_epsilon: float) -> float:
    """E[(1+epsilon) 1{epsilon<c}] for epsilon=tanh(N(mean,std))."""
    upper = min(8.0, (np.arctanh(competitor_epsilon) - latent_mean) / latent_std)
    if upper <= -8.0:
        return 0.0
    z = 0.5 * (upper + 8.0) * LEGENDRE_NODES + 0.5 * (upper - 8.0)
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    payoff = 1.0 + np.tanh(latent_mean + latent_std * z)
    return float(0.5 * (upper + 8.0) * np.sum(LEGENDRE_WEIGHTS * density * payoff))


def main() -> None:
    competitor_epsilon = 0.5
    std_values = np.geomspace(0.03, 1.2, 36)
    mean_grid = np.linspace(-0.5, 0.8, 521)
    rows = []
    for std in std_values:
        values = np.asarray([expected_spread_factor(mean, std, competitor_epsilon) for mean in mean_grid])
        best = int(values.argmax())
        rows.append(
            {
                "latent_std": std,
                "optimal_latent_mean": mean_grid[best],
                "optimal_deterministic_epsilon": np.tanh(mean_grid[best]),
                "expected_spread_factor": values[best],
            }
        )

    output = ROOT / "results" / "replication"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stochastic_persistent_optimum.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.semilogx(
        [row["latent_std"] for row in rows],
        [row["optimal_deterministic_epsilon"] for row in rows],
        marker="o",
        markersize=3,
        label="best tanh-Gaussian mean",
    )
    axis.axhline(competitor_epsilon, color="black", linestyle="--", label="deterministic 0.5⁻ benchmark")
    axis.axvline(0.72, color="tab:red", linestyle=":", label="long-run learned std ≈ 0.72")
    axis.set(xlabel="Latent Gaussian standard deviation", ylabel="Optimal deterministic epsilon", title="Persistent competitor: policy variance shifts the best response")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "stochastic_persistent_optimum.png", dpi=160)
    plt.close(figure)
    print(f"Saved analytical response curve to {output}")


if __name__ == "__main__":
    main()
