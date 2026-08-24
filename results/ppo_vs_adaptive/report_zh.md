# PPO vs Adaptive MM screening

配置：3 seeds × 98 rollouts × 1024 steps。PPO 使用固定 reference configuration；Adaptive target=0.5、gamma=2、beta=0.35。每个 checkpoint 统计此前 20,480 个真实 training-path steps。

## Lifecycle check

- `PPOAgent.train()` 只在 training run 开头调用一次 `env.reset()`；1024-step rollout boundary 不调用 reset。
- 每个 seed 实测 3 次 reset 均发生在第一条 training transition 之前（environment construction、agent initialization、training start）。产生 transition 后不再 reset，因此 Adaptive state 跨全部 98 个 rollout 持续存在。
- 每个 seed 恰有 121 个 cold-start steps，且结束时 response table 完整。

## PPO economics

下表为 3-seed mean；每行使用该 checkpoint 前 20,480 steps。

| Step | Market share | Spread PnL/step | Total PnL/step | Mean abs inventory | Inventory std | Mean hedge | Mean bid/ask epsilon |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.9711 | 0.4813 | 0.4332 | 4.640 | 5.896 | 0.492 | 0.136 / 0.131 |
| 60,416 | 0.9999 | 0.4928 | 0.4584 | 7.028 | 7.875 | 0.382 | 0.650 / 0.642 |
| 100,352 | 1.0000 | 0.9352 | 0.8966 | 9.345 | 9.542 | 0.309 | 0.773 / 0.790 |

### Final window by seed

| Seed | PPO share | Spread PnL/step | Total PnL/step | Mean abs inventory | Inventory std | Mean hedge |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.4910 | 0.4718 | 4.394 | 5.450 | 0.516 |
| 1 | 1.0000 | 0.8807 | 0.8469 | 14.700 | 13.894 | 0.182 |
| 2 | 1.0000 | 1.4339 | 1.3711 | 8.941 | 9.283 | 0.228 |

## Adaptive behavior

| Step | Market share | Mean abs inventory | Inventory std | Mean hedge | Mean base epsilon | Mean bid/ask epsilon |
|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.028923 | 0.657 | 6.592 | 0.013 | 0.960 | 0.942 / 0.942 |
| 60,416 | 0.000097 | 0.001 | 0.055 | 0.000 | 1.000 | 1.000 / 1.000 |
| 100,352 | 0.000013 | 0.000 | 0.018 | 0.333 | 1.000 | 1.000 / 1.000 |

## PPO diagnostics

| Step | Approx KL | Clip fraction | Value loss |
|---:|---:|---:|---:|
| 20,480 | 0.0055 | 0.0231 | 29254.0 |
| 60,416 | 0.0105 | 0.0493 | 23122.2 |
| 100,352 | 0.0117 | 0.0660 | 178500.3 |

## 判断

- **PPO 有部分学习信号，但不具备跨 seed 一致性。** 3-seed mean total PnL/step 从 0.4332 升至 0.8966，但 seed 0 从 0.4884 略降至 0.4718，改善主要来自 seeds 1/2。Final total PnL/step cross-seed std=0.3688。
- **Market share 的一致收敛不代表健康竞争。** PPO share 从 0.9711 升至 1.0000（final std=0.0000），原因是 Adaptive 几乎完全退出，而不是双方稳定在动态均衡。
- **库存控制变差。** PPO mean |inventory| 从 4.640 升至 9.345，inventory std 从 5.896 升至 9.542，同时 mean hedge 从 0.492 降至 0.309；seed 1 final mean |inventory| 达 14.700。
- **Adaptive 只在早期发生退让，之后没有持续跟踪 PPO。** mean base epsilon 从 0.960 变为 1.000（final std=0.000），market share 从 0.028923 降至 0.000013。60,416 steps 起三条 seed 的 base quote 均基本固定为 1.0。
- Adaptive final mean hedge=0.333 由 seed 1 在近零库存、近零成交下的 hedge fraction≈1 拉高；对应名义 hedge size 几乎为零，不能解读为有效风险控制。
- **观察到吸收态 pathology，但没有发现 lifecycle implementation bug。** 从 trajectory 和现有 decision/update 规则推断：Step 1 选择 `(1.0, 1.0)` 后，Adaptive 对 PPO 的更窄连续报价几乎没有成交；only-executed-cell update 继续以零成交刷新该 cell，而其他候选保留一次 cold-start 的旧统计。当前没有 exploration、interpolation 或周期性 re-probing，因此无法重新进入市场。这是既定 Adaptive choices 组合产生的机制性限制，不是 rollout reset 导致。
- Approx KL/clip fraction 未显示直接的 policy-update explosion，但 final value loss 在 seeds 1/2 显著增大（分别 164,470 / 352,284），与更大的 return/inventory scale 一起构成稳定性警讯。

## Go / No-Go

**No-Go for 5-seed long-run confirmation in the intended online-adapting-opponent interpretation.** 当前 screening 实际在早期后变成 PPO 对一个退出市场的 opponent；延长运行会增加样本量，但不会回答 PPO 能否在持续 online adaptation 下稳定学习。按本轮 scope 不修改 Adaptive，也不自动做 hyperparameter sweep。
