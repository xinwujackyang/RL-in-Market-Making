# PPO vs Adaptive MM with persistent diagonal probing

配置与 no-probing Phase 4 完全相同：3 seeds × 98 rollouts × 1024 steps；Adaptive target=0.5、gamma=2、beta=0.35。唯一实验差异是 cold start 后每 100 个 adaptive steps 强制一次 deterministic diagonal probe。每个 checkpoint 统计此前 20,480 个真实 training-path steps。

## Probing 与 lifecycle

- 每个 seed 仍只有 121 cold-start steps，response table 跨 rollout 持续存在。
- Probe 顺序为 epsilon grid ascending round-robin；quote 不经过 Step 2，hedge 仍使用现有 hedge objective。Response update 未改变。
- `Base=1 frequency` 只统计非 cold-start、非 probe 的正常 Step 1 decisions，避免 forced probe 机械污染该指标。

| Step | Probe steps/window | Probe fraction |
|---:|---:|---:|
| 20,480 | 203 | 0.9912% |
| 60,416 | 204 | 0.9961% |
| 100,352 | 205 | 1.0010% |

## Adaptive：no-probe vs probe

| Step | No-probe share | Probe share | No-probe mean base | Probe mean base | Probe base=1 frequency |
|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.028923 | 0.414033 | 0.960 | 0.452 | 31.134% |
| 60,416 | 0.000097 | 0.289528 | 1.000 | 0.662 | 53.881% |
| 100,352 | 0.000013 | 0.316763 | 1.000 | 0.696 | 49.243% |

## PPO：no-probe vs probe

| Step | No-probe share | Probe share | No-probe total PnL/step | Probe total PnL/step | No-probe mean abs inventory | Probe mean abs inventory |
|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.9711 | 0.5860 | 0.4332 | 0.1447 | 4.640 | 5.726 |
| 60,416 | 0.9999 | 0.7105 | 0.4584 | 0.1570 | 7.028 | 5.520 |
| 100,352 | 1.0000 | 0.6832 | 0.8966 | 0.2838 | 9.345 | 6.324 |

## PPO economics with probing

| Step | Share | Spread PnL/step | Total PnL/step | Mean abs inventory | Inventory std | Mean hedge | Mean bid/ask epsilon |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.5860 | 0.2112 | 0.1447 | 5.726 | 7.254 | 0.480 | 0.089 / 0.003 |
| 60,416 | 0.7105 | 0.1985 | 0.1570 | 5.520 | 6.693 | 0.422 | 0.185 / 0.044 |
| 100,352 | 0.6832 | 0.3408 | 0.2838 | 6.324 | 7.196 | 0.367 | 0.250 / 0.219 |

### Final window by seed

| Seed | PPO share | PPO total PnL/step | PPO mean abs inventory | Adaptive share | Adaptive mean base | Base=1 frequency |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7562 | 0.2543 | 7.899 | 0.2438 | 0.860 | 64.054% |
| 1 | 0.7411 | 0.3171 | 5.545 | 0.2589 | 0.739 | 58.451% |
| 2 | 0.5524 | 0.2798 | 5.527 | 0.4476 | 0.488 | 25.223% |

## PPO diagnostics

| Step | Approx KL | Clip fraction | Value loss |
|---:|---:|---:|---:|
| 20,480 | 0.0040 | 0.0231 | 4920.4 |
| 60,416 | 0.0023 | 0.0019 | 6740.9 |
| 100,352 | 0.0118 | 0.0607 | 1782.9 |

## 判断

- **Persistent probing 解除了原 absorbing state。** No-probe Adaptive final share=0.000013；probe 后为 0.316763（seed range 0.2438–0.4476）。三个 seed 都保持非平凡成交份额。
- **Step 1 不再锁死。** Final normal-decision base=1 frequency 从 no-probe 的 100% 降至 49.243%（cross-seed std=17.138%；seed range 25.223%–64.054%）。Adaptive mean base 和 share 在三个 checkpoints 继续变化，而非停在 1.0/0。
- **PPO 不再机械垄断。** Final share 从 no-probe 1.0000 降至 0.6832；probe screening 中 share 从 0.5860 经 0.7105 变为 0.6832，显示双方持续交互。
- **PPO 仍表现出学习，且 economics 更跨-seed 一致。** Total PnL/step 从 0.1447 升至 0.2838，3/3 seeds 均改善。Final PnL 较 no-probe 的 0.8966 更低，因为 opponent 不再退出；但 cross-seed std 从 0.3688 降至 0.0258。
- **库存风险没有出现 Phase 4 的同等恶化。** Mean abs inventory 从 5.726 变为 6.324；no-probe final 为 9.345。Final 三 seed 为 7.899, 5.545, 5.527。
- **未看到 diagonal probing 被 stale off-diagonal statistics 再次阻断。** Final Adaptive actual bid/ask 均明显低于或随 base 改变，并持续获得订单；这不证明所有 off-diagonal estimates 充分准确，但本轮没有出现由它们导致的退出态。
- Probe schedule 精确：每个 seed 共 1,002 probe steps，占全部 100,352 steps 的 0.9985%；cold start 仍为 121 steps。

## Go / No-Go

**Go for 5-seed × 204,800-step long-run confirmation.** Minimal persistent diagonal information acquisition 足以让 Adaptive 保持 behaviorally active，PPO/Adaptive 在整个 screening 中持续相互影响。Seed 间 Adaptive share/base dynamics 仍有 dispersion（final share std=0.0927），应由 long-run confirmation 判断其是否收敛，而不是在本轮继续修改 estimator。
