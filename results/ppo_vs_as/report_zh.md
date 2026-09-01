# PPO vs normalized stationary A-S

## Setup and integrity

PPO 使用冻结的 reference configuration，从 scratch 对 frozen A-S 训练：3 seeds × 204,800 steps，unit flow，sigma=0.2。每个 checkpoint 在固定但与训练独立的 10 条 20-trading-day paths（5,200 steps）上执行 deterministic PPO policy；A-S 继续使用 native winner-take-all execution。五个 checkpoint 为 20,480、60,416、100,352、150,528、204,800。所有 seed/checkpoint/agent 逐步与 aggregate PnL 恒等式均通过；A-S final side clip frequency 低于 10% gate。

## Checkpoint economics：三 seed mean（population std）

| Step | Agent | Share | Spread/step | Norm spread | Total/step | E|q| | Hedge h |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20,480 | PPO | 0.4065 ± 0.1087 | 0.1695 ± 0.0313 | 0.9744 ± 0.1108 | 0.1442 ± 0.0381 | 6.179 ± 1.261 | 0.4167 ± 0.0982 |
| 20,480 | A-S | 0.5935 ± 0.1087 | 0.2261 ± 0.0515 | 0.8589 ± 0.0549 | 0.2194 ± 0.0447 | 4.546 ± 0.447 | 0.0000 ± 0.0000 |
| 60,416 | PPO | 0.3637 ± 0.2409 | 0.1275 ± 0.0692 | 0.8580 ± 0.0912 | 0.1333 ± 0.0632 | 5.153 ± 2.351 | 0.3397 ± 0.1131 |
| 60,416 | A-S | 0.6363 ± 0.2409 | 0.2552 ± 0.1114 | 0.8791 ± 0.0886 | 0.2650 ± 0.1098 | 6.701 ± 0.886 | 0.0000 ± 0.0000 |
| 100,352 | PPO | 0.4218 ± 0.3008 | 0.1135 ± 0.0531 | 0.7431 ± 0.1906 | 0.1079 ± 0.0554 | 6.117 ± 3.305 | 0.3468 ± 0.1501 |
| 100,352 | A-S | 0.5782 ± 0.3008 | 0.2390 ± 0.1287 | 0.9159 ± 0.0490 | 0.2589 ± 0.1284 | 7.214 ± 0.209 | 0.0000 ± 0.0000 |
| 150,528 | PPO | 0.4093 ± 0.2757 | 0.1306 ± 0.0664 | 0.8071 ± 0.1301 | 0.1273 ± 0.0615 | 6.853 ± 4.221 | 0.3335 ± 0.1550 |
| 150,528 | A-S | 0.5907 ± 0.2757 | 0.2408 ± 0.1190 | 0.9011 ± 0.0595 | 0.2552 ± 0.1033 | 9.057 ± 1.466 | 0.0000 ± 0.0000 |
| 204,800 | PPO | 0.4292 ± 0.3015 | 0.1367 ± 0.0716 | 0.8130 ± 0.1336 | 0.1340 ± 0.0796 | 8.780 ± 5.801 | 0.2640 ± 0.1366 |
| 204,800 | A-S | 0.5708 ± 0.3015 | 0.2322 ± 0.1289 | 0.8889 ± 0.0720 | 0.2388 ± 0.1252 | 9.581 ± 1.972 | 0.0000 ± 0.0000 |

## Final comparison

| Metric | PPO | A-S | PPO - A-S |
|---|---:|---:|---:|
| Market share | 0.4292 ± 0.3015 | 0.5708 ± 0.3015 | -0.1415 |
| Spread PnL/step | 0.1367 ± 0.0716 | 0.2322 ± 0.1289 | -0.0955 |
| Spread/unit | 0.0179 ± 0.0030 | 0.0195 ± 0.0016 | -0.0016 |
| Normalized spread monetization | 0.8130 ± 0.1336 | 0.8889 ± 0.0720 | -0.0760 |
| Inventory PnL/step | 0.0131 ± 0.0058 | 0.0066 ± 0.0271 | 0.0065 |
| Hedge cost/step | 0.0158 ± 0.0040 | 0.0000 ± 0.0000 | 0.0158 |
| Total PnL/step | 0.1340 ± 0.0796 | 0.2388 ± 0.1252 | -0.1048 |
| E|q| | 8.780 ± 5.801 | 9.581 ± 1.972 | -0.801 |
| Inventory std | 9.937 ± 3.981 | 7.670 ± 2.057 | 2.266 |

## Policy comparison

1. **Inventory-skew direction.** Final PPO regression is `k(q) = 0.1255 + 0.0476 q` (R²=0.578); for |q|>=1, the skew has the A-S corrective sign in 96.0% of states. 因此 PPO 是否学到同方向 inventory control 可以直接从 slope/sign 判断。
2. **Linearity.** A-S pre-clip slope is exactly 0.1. PPO slope is 0.0476, and linear R²=0.578. `inventory_skew_bins.csv` and the single comparison figure retain the visible nonlinear departures.
3. **Symmetric quote level.** PPO mean-state variation std(m)=0.4659; the inventory-only fit is `m(q)=0.2813-0.0179q` (R²=0.294). A-S has fixed pre-clip m=0, so any PPO variation is an additional state-dependent margin/volume degree of freedom, though this regression alone is not causal attribution.
4. **External hedge.** PPO mean deterministic h=0.2640; the fit against |q| has slope -0.0120 (R²=0.320, intercept 0.3694). A-S h is identically zero.
5. **Operating point.** PPO final share=0.4292 and normalized spread monetization=0.8130; A-S is 0.5708 and 0.8889. This locates the two policies on different margin-volume points under identical routing.

## Economic conclusion

At 204,800 steps PPO total PnL/step is 0.1340 versus A-S 0.2388; PPO wins on 1/3 seed paths. The mean delta -0.1048 decomposes exactly into spread -0.0955 + inventory 0.0065 - hedge-cost difference 0.0158. This is a benchmark comparison under native winner-take-all execution, not evidence that current routing identifies A-S k.

## Final policy behavior by seed

| Seed | Skew intercept | Skew slope | R² | Corrective sign | std(m) | Mean h |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.0519 | 0.0747 | 0.594 | 100.0% | 0.2804 | 0.2758 |
| 1 | 0.6443 | 0.0205 | 0.348 | 100.0% | 0.2728 | 0.0910 |
| 2 | 0.4146 | 0.0284 | 0.596 | 77.6% | 0.3775 | 0.4251 |

Pooled results therefore conceal meaningful seed heterogeneity: every seed has a positive inventory-skew slope, but the corrective-sign frequency and hedge usage remain materially different. The pooled fit and figure summarize the common direction; the per-seed table is the robustness check.

## Frozen benchmark taxonomy

| Method | Type | Learns? | Inventory control | External hedge |
|---|---|---|---|---|
| Persistent | heuristic | No | None | No |
| Adaptive | online empirical | Yes, response table | quote + hedge optimization | Yes |
| A-S | stochastic control | No | analytical quote skew | No |
| PPO | model-free RL | Yes | learned quote skew | Yes |
