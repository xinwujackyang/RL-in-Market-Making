# PPO vs Adaptive MM: realized PnL decomposition

## Setup and sanity

Phase 10b artifacts do not retain both agents' per-step PnL components, so this analysis reran the identical Phase 8/10b benchmark: calibrated Adaptive MM, probe interval=20, current reference PPO, 3 seeds x 100,352 formal steps. Saved Phase 8 market share, PPO total PnL and both agents' mean absolute inventories reproduce to 1e-12. Every recorded step and every aggregate satisfy Spread + Inventory - HedgeCost = Total.

## Evolution: 3-seed means

| Step | Agent | Share | Spread/step | Spread/unit | Inventory/step | Hedge cost/step | Total/step | Mean abs inventory |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | PPO | 0.4338 | 0.1143 | 0.0132 | -0.0063 | 0.0696 | 0.0383 | 5.958 |
| 20,480 | Adaptive | 0.5662 | 0.1493 | 0.0131 | 0.0062 | 0.0609 | 0.0946 | 5.585 |
| 60,416 | PPO | 0.4054 | 0.0789 | 0.0097 | 0.0096 | 0.0420 | 0.0464 | 5.726 |
| 60,416 | Adaptive | 0.5946 | 0.1163 | 0.0098 | -0.0106 | 0.0410 | 0.0648 | 5.580 |
| 100,352 | PPO | 0.4560 | 0.1719 | 0.0192 | 0.0025 | 0.0633 | 0.1111 | 5.518 |
| 100,352 | Adaptive | 0.5440 | 0.1971 | 0.0178 | 0.0082 | 0.0633 | 0.1420 | 4.959 |

## Final by seed

| Seed | Agent | Share | Spread/step | Spread/unit | Inventory/step | Hedge cost/step | Total/step | Mean abs inventory |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Adaptive | 0.4696 | 0.0893 | 0.0095 | -0.0055 | 0.0261 | 0.0577 | 4.023 |
| 0 | PPO | 0.5304 | 0.1231 | 0.0116 | 0.0201 | 0.0306 | 0.1125 | 5.060 |
| 1 | Adaptive | 0.6044 | 0.1725 | 0.0143 | -0.0025 | 0.0567 | 0.1133 | 5.609 |
| 1 | PPO | 0.3956 | 0.1262 | 0.0160 | 0.0169 | 0.0734 | 0.0697 | 5.935 |
| 2 | Adaptive | 0.5582 | 0.3294 | 0.0295 | 0.0326 | 0.1071 | 0.2548 | 5.246 |
| 2 | PPO | 0.4418 | 0.2665 | 0.0302 | -0.0295 | 0.0860 | 0.1511 | 5.560 |

## Final comparison: 3-seed mean (population std)

| Metric | PPO | Adaptive | PPO - Adaptive |
|---|---:|---:|---:|
| Market share | 0.4560 (0.0560) | 0.5440 (0.0560) | -0.0881 |
| Spread PnL/step | 0.1719 (0.0669) | 0.1971 (0.0996) | -0.0252 |
| Spread PnL/captured unit | 0.0192 (0.0079) | 0.0178 (0.0085) | 0.0015 |
| Inventory PnL/step | 0.0025 (0.0226) | 0.0082 (0.0173) | -0.0057 |
| Hedge cost/step | 0.0633 (0.0237) | 0.0633 (0.0334) | 0.0000 |
| Total PnL/step | 0.1111 (0.0332) | 0.1420 (0.0830) | -0.0309 |
| Mean abs inventory | 5.5181 (0.3585) | 4.9593 (0.6786) | 0.5588 |

## Interpretation

- **Spread economics:** PPO has lower final market share (0.4560 vs 0.5440) and therefore lower spread PnL/step (0.1719 vs 0.1971). However, its spread per captured unit is higher (0.0192 vs 0.0178) in 3/3 seeds. Phase 10's quote widening therefore has a real per-unit monetization benefit, but it does not offset the volume disadvantage.
- **Inventory risk:** PPO mean absolute inventory is higher (5.518 vs 4.959) in 3/3 seeds. Yet PPO inventory PnL/step is lower on average (0.0025 vs 0.0082) and changes sign across seeds. There is clear extra exposure, but no positive aggregate risk-taking explanation for PPO performance.
- **Hedging:** hedge cost/step is effectively identical (0.0633 vs 0.0633). Combined with PPO's higher inventory exposure, these data do not support a hedging-efficiency advantage or reduced reliance on external hedging from cost alone.
- **Total:** PPO total PnL/step is 0.1111 vs Adaptive 0.1420, with PPO winning only 1/3 seed paths. The mean difference -0.0309 decomposes into spread -0.0252 + inventory -0.0057 - hedge-cost difference 0.0000. The main realized economic advantage belongs to Adaptive and comes from greater captured flow/spread PnL per step; PPO's narrower edge is better spread monetization per captured unit.
