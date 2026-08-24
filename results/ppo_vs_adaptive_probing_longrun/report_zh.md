# PPO vs Adaptive MM persistent-probing long-run confirmation

## Setup

5 seeds（0–4）× 200 rollouts × 1024 steps = 204,800 training steps/seed。PPO、environment 和 Adaptive 配置完全冻结自 Phase 4b；每个 checkpoint 使用 trailing 20,480-step training-path window。Phase 4b comparison 为 3-seed screening，不是 paired statistical test。

## PPO trajectory

| Step | Share | Spread PnL/step | Total PnL/step | Mean abs inventory | Inventory std | Mean hedge | Mean bid/ask epsilon |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.5691 | 0.1820 | 0.1214 | 5.820 | 7.396 | 0.485 | 0.040 / -0.010 |
| 60,416 | 0.6467 | 0.1754 | 0.1343 | 5.899 | 7.095 | 0.425 | 0.148 / -0.011 |
| 100,352 | 0.6242 | 0.3013 | 0.2319 | 6.425 | 7.361 | 0.387 | 0.231 / 0.116 |
| 150,528 | 0.7190 | 0.4467 | 0.3796 | 7.609 | 8.467 | 0.330 | 0.219 / 0.172 |
| 204,800 | 0.7327 | 0.4770 | 0.4278 | 8.281 | 8.860 | 0.270 | 0.346 / 0.308 |

## Adaptive trajectory

| Step | Share | Mean abs inventory | Inventory std | Mean hedge | Mean base | Mean bid/ask epsilon | Base=1 frequency | Probe fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.4309 | 4.790 | 9.606 | 0.245 | 0.393 | 0.161 / 0.141 | 27.761% | 0.991% |
| 60,416 | 0.3533 | 3.297 | 5.338 | 0.178 | 0.558 | 0.368 / 0.312 | 43.481% | 0.996% |
| 100,352 | 0.3758 | 3.522 | 5.482 | 0.187 | 0.610 | 0.383 / 0.341 | 39.506% | 1.001% |
| 150,528 | 0.2810 | 2.711 | 4.709 | 0.143 | 0.679 | 0.499 / 0.507 | 53.779% | 1.001% |
| 204,800 | 0.2673 | 2.621 | 4.291 | 0.151 | 0.689 | 0.529 / 0.531 | 56.357% | 1.001% |

## Final by seed

| Seed | PPO total PnL/step | PPO share | PPO mean abs inventory | Adaptive share | Adaptive mean base | Base=1 frequency |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2443 | 0.9718 | 13.209 | 0.0282 | 0.977 | 97.978% |
| 1 | 0.8794 | 0.7339 | 5.579 | 0.2661 | 0.784 | 60.173% |
| 2 | 0.4427 | 0.5002 | 7.043 | 0.4998 | 0.303 | 13.519% |
| 3 | 0.2097 | 0.5477 | 10.688 | 0.4523 | 0.427 | 22.072% |
| 4 | 0.3628 | 0.9101 | 4.887 | 0.0899 | 0.956 | 88.044% |
| **Mean** | 0.4278 | 0.7327 | 8.281 | 0.2673 | 0.689 | 56.357% |
| **Population std** | 0.2407 | 0.1881 | 3.176 | 0.1881 | 0.276 | 33.946% |

## Phase 4b @100k vs Phase 5 @204.8k

| Metric | Phase 4b final (3 seeds) | Phase 5 final (5 seeds) |
|---|---:|---:|
| PPO total PnL/step | 0.2838 | 0.4278 |
| PPO mean abs inventory | 6.324 | 8.281 |
| PPO market share | 0.6832 | 0.7327 |
| Adaptive market share | 0.3168 | 0.2673 |
| Adaptive mean base | 0.696 | 0.689 |
| Adaptive base=1 frequency | 49.243% | 56.357% |
| PPO PnL population std | 0.0258 | 0.2407 |

## PPO diagnostics

| Step | Approx KL | Clip fraction | Value loss |
|---:|---:|---:|---:|
| 20,480 | 0.0027 | 0.0165 | 3487.8 |
| 60,416 | 0.0049 | 0.0121 | 8167.6 |
| 100,352 | 0.0088 | 0.0447 | 7360.1 |
| 150,528 | 0.0045 | 0.0189 | 30834.1 |
| 204,800 | 0.0047 | 0.0227 | 72161.0 |

## Interpretation

- **Adaptive 没有在所有 seeds 中长期保持 active。** 5-seed mean share 从 100k 的 0.3758 降至 204.8k 的 0.2673；mean base=1 frequency 从 39.5% 升至 56.4%。
- **出现两个明确的 late-stage near-absorbing paths。** seed 0: share 0.244→0.101→0.028, base=1 64.1%→87.7%→98.0%；seed 4: share 0.422→0.215→0.090, base=1 34.1%→65.9%→88.0%。这说明 1% diagonal probing 对这些 paths 主要延迟、而非永久消除 lock-in。
- **另外三个 seeds 维持 nontrivial interaction，但不是单一固定点。** seed 1: 0.259→0.284→0.266；seed 2: 0.448→0.398→0.500；seed 3: 0.507→0.408→0.452。Seeds 1/2/3 分别表现为较稳定 band 或有界振荡，而 seeds 0/4 向 PPO-dominant regime 漂移，支持 multi-agent path dependence。
- **PPO economics 持续改善。** Total PnL/step mean 从 0.1214 升至 0.4278，5/5 seeds 均改善；spread PnL/step 同时从 0.1820 升至 0.4770，因此收益提升不只是 inventory mark-to-market exposure。
- **Inventory risk 有所增加但不是无界一致恶化。** Mean abs inventory 5.820→8.281，inventory std 7.396→8.860；3/5 seeds 增加。Seed 0/3 final mean abs inventory 达 13.209/10.688，是需要保留的风险警讯。
- **Seed dispersion 不可忽略。** Final PPO PnL population std=0.2407（seed range 0.2097–0.8794），Adaptive share std=0.1881，base=1 frequency std=33.9%。
- Approx KL/clip fraction 没有共同 explosion，但 final value loss range=615.4–336492.8；seed 1 的大 value loss 与高 PnL path 一起说明 return/value scale 仍高度 path-dependent。本轮按 scope 不修复。
- Execution sanity 全部通过：5/5 runs 到达 204,800 steps；每 seed cold start=121、probe steps=2,046（总步数占比 0.9990%）；无 NaN/Inf、越界 action 或 crash。

## Go / No-Go

**No-Go for freezing this setup as a stable, reproducible interaction result.** Positive PPO learning 与 3/5 active-opponent paths 值得保留，但 2/5 seeds 在 late stage 重新接近 absorbing state，且 final regime dispersion 很大。下一步应是 targeted mechanism investigation，区分 diagonal estimates 已刷新但 Step 1 仍偏向 1.0，还是 stale off-diagonal Step 2 responses / path feedback 导致退出；本轮不修改或 sweep 任何机制。
