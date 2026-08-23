# Risk-aversion comparison

Three paired seeds, persistent competitor epsilon `0.5`, stochastic evaluation, current-step `InventoryPnL^2`, and `alpha=0.01`.

| Metric | No penalty | InvPnL-squared penalty |
|---|---:|---:|
| Mean total PnL | 110.8317 | 117.9120 |
| PnL standard deviation | 79.1072 | 78.9847 |
| Inventory standard deviation | 19.3093 | 18.6152 |
| Inventory-PnL standard deviation | 4.8105 | 4.5514 |
| Mean hedge fraction | 0.4811 | 0.5062 |
| Mean market share | 0.7062 | 0.7612 |

The penalty produces modest inventory-risk reduction. The sample is too small to interpret the higher observed mean PnL as a reliable return improvement.
