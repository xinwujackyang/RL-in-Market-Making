# PPO policy behavior against calibrated Adaptive MM (probe interval 20)

## Setup

Phase 8 artifacts only contain checkpoint aggregates, so this analysis reran the identical 3-seed x 100,352-step benchmark. Calibration, RNG restoration, fresh formal environment, PPO training and Adaptive probe interval=20 are unchanged. The rerun reproduces the saved Phase 8 mean bid/ask actions to absolute tolerance 1e-12. Inventory is captured immediately before `env.step(action)`, so every regression and bin uses decision-time inventory.

## Quoting evolution

Values are 3-seed mean (population std), using trailing 20,480-step windows.

| Step | Mean epsilon_bid | Mean epsilon_ask | Symmetric level m |
|---:|---:|---:|---:|
| 20,480 | 0.0639 (0.0568) | 0.0175 (0.0280) | 0.0407 (0.0380) |
| 60,416 | 0.1971 (0.0405) | 0.0765 (0.0549) | 0.1368 (0.0132) |
| 100,352 | 0.2092 (0.0352) | 0.1388 (0.0646) | 0.1740 (0.0164) |

## Inventory skew regressions

| Seed | Step | beta_bid | beta_ask | beta_skew | R-squared skew |
|---:|---:|---:|---:|---:|---:|
| 0 | 20,480 | -0.00764 | -0.01202 | 0.00437 | 0.0014 |
| 0 | 60,416 | 0.01673 | 0.03048 | -0.01375 | 0.0130 |
| 0 | 100,352 | 0.06986 | 0.03059 | 0.03927 | 0.0961 |
| 1 | 20,480 | 0.00439 | -0.03055 | 0.03494 | 0.0783 |
| 1 | 60,416 | 0.02423 | -0.04558 | 0.06981 | 0.2360 |
| 1 | 100,352 | 0.02155 | -0.03205 | 0.05360 | 0.1608 |
| 2 | 20,480 | 0.00432 | -0.03368 | 0.03800 | 0.0953 |
| 2 | 60,416 | 0.00001 | -0.07620 | 0.07621 | 0.3091 |
| 2 | 100,352 | -0.00698 | -0.06480 | 0.05782 | 0.2034 |

Full economically classical sign pattern counts (beta_bid>0, beta_ask<0, beta_skew>0): 20,480=2/3, 60,416=2/3, 100,352=1/3.

## Hedging by decision-time inventory magnitude

Each cell is pooled 3-seed mean hedge fraction (sample count).

| Step | |z| < 2 | 2 <= |z| < 5 | 5 <= |z| < 10 | |z| >= 10 |
|---:|---:|---:|---:|---:|
| 20,480 | 0.4877 (16,872) | 0.4918 (12,289) | 0.4771 (19,779) | 0.4604 (12,500) |
| 60,416 | 0.4601 (18,418) | 0.4610 (11,881) | 0.4386 (19,252) | 0.4201 (11,889) |
| 100,352 | 0.4393 (19,358) | 0.4573 (12,311) | 0.4167 (18,801) | 0.3697 (10,970) |

## Interpretation

- **Quoting:** pooled symmetric level changes from 0.0407 early to 0.1740 final；3/3 seeds 都向更高 m 移动。PPO 随训练更愿意整体 widen quotes。
- **Inventory skew:** final beta_skew 在 3/3 seeds 中为正，但完整 classical sign pattern 只有 1/3。Seed 1 是 bid 上调 / ask 下调；seed 0 两侧都随 inventory 上调但 bid 更快；seed 2 两侧都下调但 ask 更快。因此 PPO 稳定学到了 relative skew，却没有跨 seed 学到一致的 classical two-sided decomposition。Final R-squared 仅为 0.096–0.203，inventory 对 stochastic skew 的线性解释力有限但非零。
- **Hedging:** final pooled hedge means by increasing |z| bin are 0.4393 -> 0.4573 -> 0.4167 -> 0.3697. 它们并不随 inventory magnitude 增加；极端 |z|>=10 时反而最低。
- **Overall:** PPO 学到的是“整体报价逐渐变宽 + 以 relative quote skew 响应 inventory”的控制结构。Economically sensible inventory management 只得到部分支持：quote skew 方向最终 3/3 合理，但两侧分解不一致；hedge head 没有表现出 exposure 越大、hedge 越强的经典风险控制。
