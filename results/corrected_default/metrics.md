# Corrected default runs

These results use the Jacobian-corrected bounded policy, dealer-perspective trade signs, volume-weighted market share, and the paper-coherent event timeline. Each evaluation uses 10 episodes of 20 trading days.

| Metric | Single dealer | Two dealers: random | Two dealers: persistent (0.5) |
|---|---:|---:|---:|
| Mean total PnL | 787.7963 | 174.5863 | 440.8027 |
| PnL standard deviation | 40.6519 | 71.4420 | 54.4280 |
| Mean spread PnL / step | 1.6734 | 0.4748 | 0.9488 |
| Mean inventory PnL / step | 0.0056 | 0.0711 | 0.0039 |
| Mean hedge cost / step | 0.1640 | 0.2102 | 0.1051 |
| Inventory mean | -0.4371 | -3.1102 | -0.8133 |
| Inventory standard deviation | 11.7910 | 18.0990 | 14.4538 |
| Mean epsilon bid | 0.8091 | 0.0401 | 0.2260 |
| Epsilon bid standard deviation | 0.0005 | 0.0019 | 0.0002 |
| Mean epsilon ask | 0.7305 | -0.0678 | -0.2160 |
| Epsilon ask standard deviation | 0.0007 | 0.0035 | 0.0002 |
| Mean hedge fraction | 0.6657 | 0.5246 | 0.3619 |
| Base-Gaussian entropy proxy | 4.2556 | 4.2448 | 4.2506 |
| RL market share | n/a | 0.5107 | 1.0000 |

The bounded-policy correction removes likelihood/execution inconsistency. It materially changes levels but does not by itself solve persistent-competitor learning: the bid and ask means remain asymmetric and well below the `0.5^-` deterministic spread-PnL benchmark.
