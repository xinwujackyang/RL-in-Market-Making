# PPO batch reuse 与 entropy regularization screening

## 总结

本轮完成两个彼此独立的 `3 seeds × 100,352 steps` screening，没有运行 `3 epochs + ent_coef=0` 的 combined condition。

- **Part A — Reduced batch reuse：Promising。** 将 PPO epochs 从 `10` 降到 `3` 后，final aggregate two-side MAE 从 `0.1385` 降至 `0.0670`（下降 `51.6%`），seed dispersion 从 `0.0533` 降至 `0.0290`（下降 `45.6%`），worst-side 从 `0.2982` 降至 `0.2002`（下降 `32.9%`），且 `3/3` paired seeds 改善。Final approx KL 和 clip fraction 也显著下降，policy quality 同时改善，满足预先规定的 promising 条件。
- **Part B — No entropy bonus：No-Go。** 将 `ent_coef` 从 `0.003` 降到 `0` 后，final MAE 小幅改善至 `0.1249`，但 seed dispersion 增至 `0.0636`（上升 `19.3%`），worst-side 增至 `0.3063`，结果 mixed。State-dependent exploration 没有 collapse，但不能据此正式移除 entropy bonus。

本轮按 screening 停止，不自动扩展到 `5 seeds × 204,800 steps`。

## Controlled setup

统一 baseline 是 state-dependent squashed Gaussian、`n_epochs=10`、`ent_coef=0.003`，直接复用 `results/state_dependent_std/training_trajectories.csv` 中 seeds 0–2 的相同 checkpoints，不重跑 baseline。

两个新 condition 分别只改变一个变量：

- `epochs_3`：只将 `n_epochs` 从 `10` 改为 `3`，保留 `ent_coef=0.003`。
- `entropy_0`：只将 `ent_coef` 从 `0.003` 改为 `0`，保留 `n_epochs=10`。

其他 environment、observation、network 和 PPO 配置完全保持 current baseline。Seed dispersion 是 3 个 per-seed two-side MAE 的 population standard deviation；worst-side 是 3 seeds × 2 sides 中最大的绝对 deterministic epsilon。

## Part A — Reduced batch reuse

研究问题：同一 rollout 重复优化 10 epochs 是否过强，减少到 3 epochs 能否改善 PPO stability？

| Step | Baseline MAE | 3-epoch MAE | Baseline dispersion | 3-epoch dispersion | Baseline worst-side | 3-epoch worst-side | Paired wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,480 | 0.0832 | 0.0683 | 0.0385 | 0.0444 | 0.1801 | 0.1524 | 2/3 |
| 60,416 | 0.1478 | 0.0735 | 0.0329 | 0.0163 | 0.3828 | 0.1100 | 3/3 |
| 100,352 | 0.1385 | 0.0670 | 0.0533 | 0.0290 | 0.2982 | 0.2002 | 3/3 |

20k 时 3-epoch dispersion 略高，但 MAE 与 worst-side 已更低；60k 和 100k 时 MAE、dispersion、worst-side 全部改善，并且 3 个 seeds 全部优于 paired baseline。Trajectory 没有显示 late screening pathology。

### Policy movement diagnostics

| Step | Baseline approx KL | 3-epoch approx KL | Baseline clip fraction | 3-epoch clip fraction |
| ---: | ---: | ---: | ---: | ---: |
| 20,480 | 0.01053 | 0.00642 | 0.0483 | 0.0187 |
| 60,416 | 0.00521 | 0.00602 | 0.0153 | 0.0214 |
| 100,352 | 0.00887 | 0.00069 | 0.0321 | 0.0000 |

60k checkpoint 的 KL / clipping 没有低于 baseline，但 final checkpoint 的 approx KL 下降约 `92.2%`，clip fraction 从 `3.21%` 降到 `0`。更重要的是，较小 policy movement 与更好的 MAE/stability 同时出现，而不是 diagnostic 单独变漂亮。

### Final paired seeds

| Seed | Baseline bid | Baseline ask | Baseline MAE | 3-epoch bid | 3-epoch ask | 3-epoch MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1161 | 0.2982 | 0.2071 | 0.0182 | 0.1180 | 0.0681 |
| 1 | -0.1155 | 0.0389 | 0.0772 | 0.0103 | -0.0517 | 0.0310 |
| 2 | 0.1176 | 0.1445 | 0.1310 | 0.2002 | 0.0037 | 0.1020 |

**Part A decision：Promising。** 结果支持“10 epochs 的 batch reuse 对当前 setup 可能过强”。`n_epochs=3` 值得在后续单独做 `5 seeds × 204,800 steps` confirmation，但本轮不自动执行 full run。

## Part B — No entropy bonus

研究问题：去掉显式 entropy incentive 后，policy learning 是否受损，以及 state-dependent exploration 是否仍然存在？

| Step | Baseline MAE | Entropy-0 MAE | Baseline dispersion | Entropy-0 dispersion | Baseline worst-side | Entropy-0 worst-side | Paired wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,480 | 0.0832 | 0.0564 | 0.0385 | 0.0199 | 0.1801 | 0.1157 | 3/3 |
| 60,416 | 0.1478 | 0.1016 | 0.0329 | 0.0273 | 0.3828 | 0.1793 | 2/3 |
| 100,352 | 0.1385 | 0.1249 | 0.0533 | 0.0636 | 0.2982 | 0.3063 | 2/3 |

Entropy-0 在早期 checkpoints 表现较好，但到 100k 时优势减弱：aggregate MAE 仅下降 `9.8%`，seed dispersion 上升 `19.3%`，worst-side 上升 `2.7%`。Seed 2 明显改善，但 seeds 0 和 1 保留较大 offset，结果不是一致的 stability improvement。

### State-dependent exploration

| 100,352-step statistic | Baseline | Entropy-0 | Change |
| --- | ---: | ---: | ---: |
| Latent sigma bid mean | 0.9520 | 0.9314 | -2.2% |
| Latent sigma ask mean | 0.9502 | 0.9324 | -1.9% |
| Bid sigma state std | 0.2060 | 0.1756 | -14.8% |
| Ask sigma state std | 0.1502 | 0.1107 | -26.3% |

去掉 entropy bonus 后，latent exploration scale 只温和下降；更重要的是，3 个 seeds 的 bid/ask sigma state std 均大于零。Final stochastic evaluation 中 sampled epsilon std 仍为 bid `0.6197`、ask `0.6827`。因此 state-dependent exploration 没有 collapse，policy gradient 本身足以维持明显的 state-dependent variance structure。

Matched baseline 在 100,352 checkpoint 没有保存 sampled-action std，所以这里不把 entropy-0 sampled std 与 baseline 的 204,800-step sampled std 做跨预算比较。Entropy 绝对值也只按 latent Gaussian entropy proxy 理解，不解释为 action-space entropy。

### Final paired seeds

| Seed | Baseline bid | Baseline ask | Baseline MAE | Entropy-0 bid | Entropy-0 ask | Entropy-0 MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1161 | 0.2982 | 0.2071 | 0.1167 | 0.2371 | 0.1769 |
| 1 | -0.1155 | 0.0389 | 0.0772 | -0.3063 | -0.0187 | 0.1625 |
| 2 | 0.1176 | 0.1445 | 0.1310 | 0.0425 | -0.0282 | 0.0353 |

**Part B decision：No-Go。** Entropy bonus 不是 state-dependent variance 存在的唯一原因，但移除它没有带来稳定、跨 seed 的 policy-quality improvement。保留当前 `ent_coef=0.003`；不做 entropy sweep。

## Secondary economic sanity

新 conditions 的 final stochastic evaluation 三 seed 均值如下，仅作为 sanity check，不作为 screening 决策依据：

| Condition | Market share | Spread PnL / step | Total PnL / step | Inventory std |
| --- | ---: | ---: | ---: | ---: |
| `epochs_3` | 0.4840 | 0.1191 | 0.0807 | 9.0015 |
| `entropy_0` | 0.4940 | 0.1232 | 0.0781 | 8.2388 |

## Validation

- Existing unit tests：10/10 passed。
- Python compile：passed。
- 两个 condition 的 short PPO smoke：passed；PPO diagnostics 与 latent-sigma fields 均 finite。
- 6 个正式 runs 全部完成，无 NaN/Inf 或缺失 checkpoint。
