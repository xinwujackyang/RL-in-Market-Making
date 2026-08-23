# Phase 1 baseline snapshot

This is a read-only transcription of the corrected five-seed baseline recorded before the policy-variance investigation. Source artifacts under `results/replication/`, `results/risk_aversion/`, and `docs/replication_findings.md` were not overwritten.

| Metric | Random `U[-1,1]` | Persistent `0.5` | Persistent spread-only |
|---|---:|---:|---:|
| Sampled epsilon bid mean across seeds | 0.0728 +/- 0.1352 | 0.0260 +/- 0.1597 | -0.0310 +/- 0.0452 |
| Sampled epsilon ask mean across seeds | 0.1290 +/- 0.1853 | 0.0535 +/- 0.1520 | -0.0551 +/- 0.0366 |
| Within-policy sampled bid std | 0.6206 | 0.6199 | 0.6220 |
| Within-policy sampled ask std | 0.6074 | 0.6171 | 0.6199 |
| Latent bid std | 0.9949 | 0.9955 | 0.9710 |
| Latent ask std | 0.9987 | 0.9951 | 0.9728 |
| Mean spread PnL per step | 0.2685 | 0.4308 | 0.4591 |
| Mean total PnL | 51.4682 | 132.4854 | 122.0831 |
| Mean market share | 0.4462 | 0.6767 | 0.7354 |
| Mean inventory std | 19.0173 | 19.0733 | 18.7574 |

The baseline inventory-conditional plot shows wrong-sign aggregate skew for the random competitor and weak curves for persistent/spread-only settings. The new investigation compares deterministic and stochastic curves separately rather than modifying this baseline.
