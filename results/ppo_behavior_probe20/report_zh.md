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

## Quote decomposition regressions

Here beta_m is inventory sensitivity of the overall symmetric quote level; beta_k is inventory sensitivity of relative liquidation skew.

| Seed | Step | beta_m | beta_k | R-squared m | R-squared k |
|---:|---:|---:|---:|---:|---:|
| 0 | 20,480 | -0.00983 | 0.00437 | 0.0282 | 0.0014 |
| 0 | 60,416 | 0.02361 | -0.01375 | 0.1459 | 0.0130 |
| 0 | 100,352 | 0.05022 | 0.03927 | 0.3739 | 0.0961 |
| 1 | 20,480 | -0.01308 | 0.03494 | 0.0478 | 0.0783 |
| 1 | 60,416 | -0.01067 | 0.06981 | 0.0296 | 0.2360 |
| 1 | 100,352 | -0.00525 | 0.05360 | 0.0076 | 0.1608 |
| 2 | 20,480 | -0.01468 | 0.03800 | 0.0600 | 0.0953 |
| 2 | 60,416 | -0.03809 | 0.07621 | 0.3000 | 0.3091 |
| 2 | 100,352 | -0.03589 | 0.05782 | 0.2799 | 0.2034 |

Positive inventory-dependent relative skew counts (beta_k>0): 20,480=3/3, 60,416=2/3, 100,352=3/3.

## Hedging by decision-time inventory magnitude

Pooled across 3 seeds. Quantities use decision-time absolute inventory.

| Step | Inventory bin | Hedge fraction | Absolute hedge qty | Residual qty | N |
|---:|---|---:|---:|---:|---:|
| 20,480 | |z| < 2 | 0.4877 | 0.3470 | 0.3647 | 16,872 |
| 20,480 | 2 <= |z| < 5 | 0.4918 | 1.6823 | 1.7407 | 12,289 |
| 20,480 | 5 <= |z| < 10 | 0.4771 | 3.5898 | 3.9518 | 19,779 |
| 20,480 | |z| >= 10 | 0.4604 | 5.9144 | 7.1105 | 12,500 |
| 60,416 | |z| < 2 | 0.4601 | 0.2932 | 0.3566 | 18,418 |
| 60,416 | 2 <= |z| < 5 | 0.4610 | 1.5824 | 1.8513 | 11,881 |
| 60,416 | 5 <= |z| < 10 | 0.4386 | 3.2844 | 4.2489 | 19,252 |
| 60,416 | |z| >= 10 | 0.4201 | 5.3749 | 7.5780 | 11,889 |
| 100,352 | |z| < 2 | 0.4393 | 0.2816 | 0.3624 | 19,358 |
| 100,352 | 2 <= |z| < 5 | 0.4573 | 1.5599 | 1.8656 | 12,311 |
| 100,352 | 5 <= |z| < 10 | 0.4167 | 3.1131 | 4.4146 | 18,801 |
| 100,352 | |z| >= 10 | 0.3697 | 4.6862 | 8.3371 | 10,970 |

## Interpretation

- **Quoting:** pooled symmetric level changes from 0.0407 early to 0.1740 final；3/3 seeds 都向更高 m 移动。PPO 随训练更愿意整体 widen quotes。
- **Quote decomposition:** beta_k 最终在 3/3 seeds 中为正，说明 PPO 一致学到了 inventory-dependent relative liquidation skew。该 sign 在 early 已是 3/3、mid 短暂变为 2/3、final 回到 3/3，因此不是简单的单调 emergence。Final beta_m 则跨 seed 分别为 0.05022, -0.00525, -0.03589；inventory 同时引发的 overall quote-level adjustment 并不一致，因此不应以 bid/ask absolute slopes 必须一正一负作为主要评价标准。Final R-squared k 为 0.096–0.203，inventory 对 stochastic skew 的线性解释力有限但非零。
- **Hedge fraction:** final means by increasing |z| bin are 0.4393 -> 0.4573 -> 0.4167 -> 0.3697；fraction 本身没有随 exposure 增加。
- **Absolute hedge control:** final E[h|z|] is 0.2816 -> 1.5599 -> 3.1131 -> 4.6862；absolute hedge quantity 随 inventory magnitude 单调增加。
- **Residual exposure:** final E[(1-h)|z|] is 0.3624 -> 1.8656 -> 4.4146 -> 8.3371；residual exposure 仍随 inventory magnitude 单调增加。
- **Corrected conclusion:** PPO 的 control policy 可分解为 overall quote adjustment + 3/3 consistent inventory-dependent relative skew + absolute hedge control。E[h|z|] 随 exposure 单调增长，支持其 executed actions 包含显式 balance-sheet risk reduction；但 residual quantity 同时大幅增长，说明 PPO 不会在比例上完全 neutralize inventory risk。
