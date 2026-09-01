# Normalized stationary A-S benchmark

## 定义

Canonical A-S 保留 `r = S - q gamma V` 与 `Delta = gamma V + 2/gamma log(1 + gamma/k)`，其中 `V = P^2 sigma^2 dt H`。正式 benchmark 固定 normalized coordinates `rho=gamma D`、`kappa=kD`，而不是从当前 winner-take-all routing 拟合不存在的 smooth execution elasticity。

- risk_horizon_steps = 26（一个 6.5 小时交易日）
- neutral_epsilon = 0
- inventory_anchor = 20
- hedge_fraction = 0
- rho = 1.5246e-05
- kappa = 1.02563340266
- P0 下等价 gamma = 0.000693，k = 46.6197001209

这是 stationary scale-normalized A-S implementation：normalized control parameters 固定，其 dollar-coordinate equivalents 随 prevailing unit reference spread 缩放。

## Compatibility choices

- fixed receding one-day risk horizon，不使用训练 horizon；
- GBM local Brownian variance approximation；
- zero-inventory quote 对齐 simulator unit-flow reference quote；
- inventory_anchor 来自每步 20 units 的市场流量，而非 PPO/PnL calibration；
- native winner-take-all routing、native price/RNG/PnL accounting；
- external hedge 固定为零；
- executable epsilon 按 simulator action domain 裁剪，并显式记录 raw quote 与 clip。

## Standalone setup

3 seeds × 100,352 steps，unit flow，sigma=0.2。A-S 与 Persistent(epsilon_bid=epsilon_ask=0, hedge=0) 共享同一个 TwoDealerMarketEnv、investor orders 与 price path。

## 三 seed 结果（population std）

| Metric | A-S | Persistent |
|---|---:|---:|
| Market share | 0.4998 ± 0.0006 | 0.5002 ± 0.0006 |
| Spread PnL/step | 0.1686 ± 0.0655 | 0.2250 ± 0.0875 |
| Spread/unit | 0.0169 ± 0.0066 | 0.0225 ± 0.0087 |
| Normalized spread monetization | 0.7499 ± 0.0002 | 1.0000 ± 0.0000 |
| Inventory PnL/step | 0.0035 ± 0.0017 | -0.3278 ± 0.4636 |
| Hedge cost/step | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Total PnL/step | 0.1721 ± 0.0669 | -0.1028 ± 0.3813 |
| E|q| | 4.998 ± 0.006 | 1131.933 ± 864.701 |
| std(q) | 5.912 ± 0.007 | 621.310 ± 370.218 |

## Inventory dynamics and clipping

- E[Delta q | q>0] = -9.9962 ± 0.0142 < 0。
- E[Delta q | q<0] = 10.0010 ± 0.0208 > 0。
- Bid clip frequency = 0.0000 ± 0.0000。
- Ask clip frequency = 0.0000 ± 0.0000。
- Pooled side clip frequency = 0.0000 ± 0.0000；低于 10% 的预设理想阈值，保留 inventory_anchor=20。

两项 conditional drift 的符号直接确认 analytical skew 在 native execution 中会清算库存。所有逐步数据均满足 Spread + Inventory - HedgeCost = Total，所有统计量 finite。Standalone gate 通过后，上述 A-S 参数冻结用于 PPO 对比。
