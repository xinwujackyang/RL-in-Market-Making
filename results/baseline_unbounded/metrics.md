# Baseline: Gaussian sampling plus environment clipping

These are the unchanged-default results captured before the bounded-policy and simulator sign/timing corrections. Each evaluation uses 10 episodes of 20 trading days.

| Metric | Single dealer | Two dealers: random | Two dealers: persistent (0.5) |
|---|---:|---:|---:|
| Mean total PnL | 854.9543 | 130.8648 | 265.1032 |
| PnL standard deviation | 43.2066 | 106.0451 | 45.1347 |
| Mean spread PnL / step | 1.7900 | 0.4742 | 0.6584 |
| Mean inventory PnL / step | 0.0052 | -0.0514 | 0.0053 |
| Mean hedge cost / step | 0.1510 | 0.1711 | 0.1539 |
| Inventory mean | -0.1935 | 1.0067 | -0.1816 |
| Inventory standard deviation | 4.8591 | 11.3494 | 4.6510 |
| Mean epsilon bid | 0.8868 | 0.0465 | -0.3715 |
| Epsilon bid standard deviation | 0.0001 | 0.0153 | 0.0001 |
| Mean epsilon ask | 0.8988 | -0.0037 | -0.2350 |
| Epsilon ask standard deviation | 0.0001 | 0.0112 | 0.0001 |
| Mean hedge fraction | 0.5993 | 0.4131 | 0.6139 |
| RL market share | n/a | 0.4902 | 1.0000 |

The baseline evaluator used deterministic policy means, so these epsilon standard deviations measure state dependence rather than sampled policy variance.
