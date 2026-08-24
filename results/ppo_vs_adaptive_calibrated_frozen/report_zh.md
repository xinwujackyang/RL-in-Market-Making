# Frozen PPO vs calibrated Adaptive MM

## Setup

每 seed 先按 Phase 6 执行 10 x 121 = 1,210 calibration steps，再在 fresh formal environment 中以同一个随机初始化 theta0 stochastic policy 手写运行 100,352 steps。Formal phase 不调用 `train()`、optimizer、rollout buffer、GAE 或 PPO loss；Adaptive 参数、warm-start table、interval=100 diagonal probing 均与 Phase 6 相同。Checkpoint 指标统计此前 20,480 steps。

## Frozen trajectories

| Step | Adaptive share | Adaptive mean base | Adaptive base=1 | PPO share | PPO total PnL/step | PPO mean abs inventory |
|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.3676 | 0.484 | 40.084% | 0.6324 | 0.1603 | 5.448 |
| 60,416 | 0.4627 | 0.359 | 26.067% | 0.5373 | 0.0832 | 5.634 |
| 100,352 | 0.4004 | 0.442 | 34.977% | 0.5996 | 0.1581 | 5.548 |

## Learning vs frozen: final matched comparison

| Seed | Learning PPO final Adaptive share | Frozen PPO final Adaptive share | Learning base=1 | Frozen base=1 |
|---:|---:|---:|---:|---:|
| 0 | 0.2244 | 0.3458 | 64.962% | 44.158% |
| 1 | 0.1144 | 0.4575 | 85.983% | 26.841% |
| 2 | 0.4742 | 0.3979 | 18.969% | 33.933% |

## Conclusion

**PPO nonstationarity 是主要问题。** Frozen theta0 的三个 seed 在全部 checkpoints 中，Adaptive share 保持在 0.296–0.482，base=1 frequency 保持在 24.1%–50.5%；没有出现随时间单向 drift 到 share≈0 / base=1≈100% 的 lock-in。相比之下，learning PPO 的 seed 1 final Adaptive share=0.114、base=1=86.0%，seed 0 也明显弱于 matched frozen run。Seed 2 的 frozen result 略弱于 learning result，说明 path dispersion 没有消失，但结果不支持 Adaptive 自身 dynamics 单独造成此前的极端退出态。

PPO formal parameter max change 对全部 seed 均为 0；optimizer state、rollout buffer pointer 与 training history 也保持为空。
