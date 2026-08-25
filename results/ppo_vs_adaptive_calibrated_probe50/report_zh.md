# Calibrated Adaptive MM: probe interval 50 boundary check

## Setup and sanity

Phase 8 calibrated + learning-PPO setup 原样复用，唯一 experimental change 是 probe interval 20 -> 50。3 seeds x 100,352 formal steps；每 seed calibration=1,210、cells=121、formal cold start=0、formal probes=2,007（约 2%）。

## Interval=50 trajectories

| Seed | Adaptive share: 20k -> 60k -> 100k | Mean base: 20k -> 60k -> 100k | Base=1: 20k -> 60k -> 100k |
|---:|---:|---:|---:|
| 0 | 0.4929 -> 0.3469 -> 0.2502 | 0.331 -> 0.626 -> 0.831 | 18.60% -> 44.13% -> 62.88% |
| 1 | 0.4850 -> 0.4380 -> 0.3879 | 0.302 -> 0.514 -> 0.622 | 20.12% -> 29.47% -> 37.11% |
| 2 | 0.5269 -> 0.4571 -> 0.4673 | 0.341 -> 0.544 -> 0.474 | 15.14% -> 24.69% -> 18.50% |

## Final matched comparison

| Seed | Share @100 | Share @50 | Share @20 | Base=1 @100 | Base=1 @50 | Base=1 @20 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2244 | 0.2502 | 0.4696 | 64.962% | 62.885% | 26.069% |
| 1 | 0.1144 | 0.3879 | 0.6044 | 85.983% | 37.110% | 1.593% |
| 2 | 0.4742 | 0.4673 | 0.5582 | 18.969% | 18.500% | 7.540% |

## Final dispersion and PPO economics

| Metric | Interval=100 | Interval=50 | Interval=20 |
|---|---:|---:|---:|
| Adaptive share std | 0.1505 | 0.0897 | 0.0560 |
| Adaptive base=1 std | 27.984% | 18.198% | 10.423% |
| PPO share | 0.7290 | 0.6315 | 0.4560 |
| PPO total PnL/step | 0.3201 | 0.2440 | 0.1111 |
| PPO mean abs inventory | 7.1121 | 5.6420 | 5.5181 |

## Stop rule

Interval=50 的 PPO final mean share=0.6315、total PnL/step=0.2440、mean abs inventory=5.642，三者均位于 interval=100 与 interval=20 之间。

**Interval=50 fails；freeze probe interval=20。** Seeds 0, 1 的 Adaptive share 在三个 checkpoints 单调下降，同时 base=1 单调上升；尤其 seed 0 final share=0.250 / base=1=62.9%，已接近 interval=100 的 0.224 / 65.0% pathology。虽然 final cross-seed dispersion 居中且数值上更靠近 interval=20，这不能覆盖明确的 within-seed lock-in trajectory。不再尝试其他 probing intervals；下一阶段回到 PPO behavior analysis。
