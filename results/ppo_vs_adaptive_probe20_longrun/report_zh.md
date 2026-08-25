# PPO vs 稳定 Adaptive MM：204,800-step confirmation

## 设置与完整性检查

除 formal horizon 外，Phase 11 设置保持不变：3 seeds x 200 rollouts x 1,024 = 204,800 steps。每个 seed 使用 1,210 calibration steps、完整 121 cells、formal cold start=0、probe interval=20（10,240 probe steps，严格为 5%）。100,352 checkpoint 在绝对误差 1e-12 内复现 Phase 8/11 指标；每一步均满足 Spread + Inventory - HedgeCost = Total。

## PPO 轨迹：3-seed mean（population std）

| Step | Share | Spread/unit | Spread/step | Total PnL/step | 平均绝对库存 | Mean m |
|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.4338 (0.0142) | 0.0132 (0.0016) | 0.1143 (0.0105) | 0.0383 (0.0103) | 5.958 (0.046) | 0.0407 (0.0380) |
| 60,416 | 0.4054 (0.0262) | 0.0097 (0.0028) | 0.0789 (0.0254) | 0.0464 (0.0249) | 5.726 (0.260) | 0.1368 (0.0132) |
| 100,352 | 0.4560 (0.0560) | 0.0192 (0.0079) | 0.1719 (0.0669) | 0.1111 (0.0332) | 5.518 (0.358) | 0.1740 (0.0164) |
| 150,528 | 0.4674 (0.0298) | 0.0373 (0.0191) | 0.3379 (0.1632) | 0.2541 (0.1360) | 5.682 (0.196) | 0.2615 (0.0972) |
| 204,800 | 0.5416 (0.0504) | 0.0554 (0.0413) | 0.5854 (0.4486) | 0.4784 (0.3906) | 5.485 (0.541) | 0.3788 (0.1891) |

## Adaptive 轨迹：3-seed mean（population std）

| Step | Share | Spread/unit | Spread/step | Total PnL/step |
|---:|---:|---:|---:|---:|
| 20,480 | 0.5662 (0.0142) | 0.0131 (0.0020) | 0.1493 (0.0268) | 0.0946 (0.0309) |
| 60,416 | 0.5946 (0.0262) | 0.0098 (0.0030) | 0.1163 (0.0318) | 0.0648 (0.0197) |
| 100,352 | 0.5440 (0.0560) | 0.0178 (0.0085) | 0.1971 (0.0996) | 0.1420 (0.0830) |
| 150,528 | 0.5326 (0.0298) | 0.0305 (0.0154) | 0.3337 (0.1772) | 0.2197 (0.1242) |
| 204,800 | 0.4584 (0.0504) | 0.0411 (0.0293) | 0.3884 (0.2709) | 0.2985 (0.2421) |

## 204,800-step 最终比较

| 指标 | PPO | Adaptive | PPO - Adaptive |
|---|---:|---:|---:|
| 市场份额 | 0.5416 (0.0504) | 0.4584 (0.0504) | 0.0832 |
| Spread PnL/step | 0.5854 (0.4486) | 0.3884 (0.2709) | 0.1970 |
| Spread PnL/captured unit | 0.0554 (0.0413) | 0.0411 (0.0293) | 0.0144 |
| Inventory PnL/step | -0.0004 (0.0043) | 0.0166 (0.0326) | -0.0170 |
| Hedge cost/step | 0.1066 (0.0644) | 0.1065 (0.0667) | 0.0001 |
| Total PnL/step | 0.4784 (0.3906) | 0.2985 (0.2421) | 0.1799 |
| 平均绝对库存 | 5.4851 (0.5408) | 4.1906 (0.7260) | 1.2945 |

## 最终结果（按 seed）

| Seed | Agent | 市场份额 | Total PnL/step |
|---:|---|---:|---:|
| 0 | Adaptive | 0.3968 | 0.0364 |
| 0 | PPO | 0.6032 | 0.1077 |
| 1 | Adaptive | 0.5203 | 0.2388 |
| 1 | PPO | 0.4797 | 0.3092 |
| 2 | Adaptive | 0.4580 | 0.6203 |
| 2 | PPO | 0.5420 | 1.0183 |

## 结论

**分类：Case A。** 从 100,352 到 150,528 再到 204,800，PPO 市场份额为 0.4560 -> 0.4674 -> 0.5416, 同时每单位 captured volume 的 spread PnL 为 0.0192 -> 0.0373 -> 0.0554。PPO 最终收回成交量，同时没有放弃其较高的单位成交 realized margin。

PPO total PnL/step 上升为 0.1111 -> 0.2541 -> 0.4784；mean symmetric quote level 上升为 0.1740 -> 0.2615 -> 0.3788。最终 checkpoint 中，PPO 在 3/3 条 seed 路径上取得更高 total PnL。其平均 total-PnL 优势 0.1799 分解为 spread 0.1970 + inventory -0.0170 - hedge-cost 差 0.0001；优势来自 spread/margin-volume，而不是更高的 inventory PnL 或更低的 hedging cost。

**尚未出现 plateau。** 到 204,800 时，share、quote level、spread/unit 和 total PnL 仍在显著变化，同时 late training 的 cross-seed dispersion 急剧扩大（PPO total-PnL std 0.0332 -> 0.1360 -> 0.3906）。Long-run evidence 支持更好的 PPO economics regime，但不支持 policy/economic convergence。按照 stop rule，实验在 204,800 停止，不自动继续训练。
