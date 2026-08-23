# Five-seed best-response replication summary

Metrics below evaluate sampled policy actions over 5,200 steps per seed. Values are mean across seeds; `+/-` is the population standard deviation across seed-level means.

| Metric | Random `U[-1,1]` | Persistent `0.5` | Persistent, spread-only |
|---|---:|---:|---:|
| Sampled epsilon bid | 0.0728 +/- 0.1352 | 0.0260 +/- 0.1597 | -0.0310 +/- 0.0452 |
| Sampled epsilon ask | 0.1290 +/- 0.1853 | 0.0535 +/- 0.1520 | -0.0551 +/- 0.0366 |
| Within-policy epsilon bid std | 0.6206 | 0.6199 | 0.6220 |
| Within-policy epsilon ask std | 0.6074 | 0.6171 | 0.6199 |
| Mean total PnL | 51.4682 | 132.4854 | 122.0831 |
| PnL std | 103.5131 | 83.6827 | 10.7851 |
| Mean spread PnL / step | 0.2685 | 0.4308 | 0.4591 |
| Mean market share | 0.4462 | 0.6767 | 0.7354 |
| Mean latent bid std | 0.9949 | 0.9955 | 0.9710 |
| Mean latent ask std | 0.9987 | 0.9951 | 0.9728 |

The random result is near epsilon zero only after averaging unstable seeds, so it does not meet a reliable-concentration criterion. The persistent spread-only result rules out inventory and hedge terms as the main cause of the deterministic-benchmark gap. A 204,800-step spread-only run reduced latent standard deviations to about `0.72` but retained means near zero, consistent with the separately computed stochastic-policy optimum.
