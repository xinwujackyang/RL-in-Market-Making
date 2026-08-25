# Calibrated Adaptive MM: probe interval 20

## Setup and sanity

Phase 6 calibrated + learning-PPO setup 原样复用。唯一 experimental change 是 deterministic diagonal round-robin probe interval 100 -> 20；3 seeds x 100,352 formal steps，checkpoint 使用 trailing 20,480-step window。每 seed calibration=1,210、cells=121、formal cold start=0、formal probes=5,017（4.9994%）；每个 checkpoint window 含 1,024 probes（5.000%）。

## Adaptive trajectory

| Step | Share mean (std) | Mean base | Base=1 mean (std) | Probe fraction |
|---:|---:|---:|---:|---:|
| 20,480 | 0.5662 (0.0142) | 0.206 | 4.816% (1.209%) | 5.000% |
| 60,416 | 0.5946 (0.0262) | 0.308 | 3.641% (1.571%) | 5.000% |
| 100,352 | 0.5440 (0.0560) | 0.413 | 11.734% (10.423%) | 5.000% |

| Seed | Adaptive share: 20k -> 60k -> 100k | Base=1: 20k -> 60k -> 100k |
|---:|---:|---:|
| 0 | 0.5666 -> 0.6276 -> 0.4696 | 5.57% -> 4.03% -> 26.07% |
| 1 | 0.5486 -> 0.5926 -> 0.6044 | 5.77% -> 1.55% -> 1.59% |
| 2 | 0.5834 -> 0.5636 -> 0.5582 | 3.11% -> 5.34% -> 7.54% |

## Final matched comparison

| Seed | Adaptive share, int=100 | Adaptive share, int=20 | Base=1, int=100 | Base=1, int=20 |
|---:|---:|---:|---:|---:|
| 0 | 0.2244 | 0.4696 | 64.962% | 26.069% |
| 1 | 0.1144 | 0.6044 | 85.983% | 1.593% |
| 2 | 0.4742 | 0.5582 | 18.969% | 7.540% |

## Final dispersion and PPO economics

| Metric | Interval=100 | Interval=20 |
|---|---:|---:|
| Adaptive share std | 0.1505 | 0.0560 |
| Adaptive base=1 std | 27.984% | 10.423% |
| PPO share | 0.7290 | 0.4560 |
| PPO total PnL/step | 0.3201 | 0.1111 |
| PPO mean abs inventory | 7.112 | 5.518 |

## Conclusion and next step

PPO final mean share 0.7290 -> 0.4560，total PnL/step 0.3201 -> 0.1111，mean abs inventory 7.112 -> 5.518。更频繁 probing 让 Adaptive 保持竞争，因此 PPO 的份额与 PnL 明显降低，同时 inventory exposure 下降。

**Tracking-bandwidth hypothesis 得到支持。** Seed 1 的退出态消失，三个 seeds 均保持 meaningful interaction，且 Adaptive share/base=1 的 cross-seed dispersion 同时下降。Seed 0 late window 有一定回落，但没有接近此前 lock-in。下一步应测试能否以更有选择性的 refresh allocation 降低 5% probing cost，而不是进入 Step 1/Step 2 mechanism diagnosis。
