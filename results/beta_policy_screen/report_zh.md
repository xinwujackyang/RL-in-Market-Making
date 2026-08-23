# Beta bounded-action policy screening

## 结论

本轮结论为 **No-Go**。在 `3 seeds × 100,352 steps` 的 paired screening 中，Beta policy 没有比现有 state-dependent squashed Gaussian 显示出清晰优势：final aggregate two-side MAE 从 `0.1385` 变为 `0.1395`（高 `0.8%`），MAE seed dispersion 从 `0.0533` 变为 `0.0608`（高 `14.1%`）。

Beta 的 final worst-side error 从 `0.2982` 降至 `0.2552`，且 `2/3` paired seeds 的 MAE 改善；但 primary MAE 没有改善、seed dispersion 反而变差，训练中段结果也较 mixed。因此不满足预先规定的 Go 条件。本轮在 screening 后停止，不扩到 `5 seeds × 204,800 steps`，也不做 concentration 或 entropy sweep。

## Controlled setup

唯一实验变量是 action distribution：

- Baseline：state-dependent squashed Gaussian；直接复用 `results/state_dependent_std/training_trajectories.csv` 中 seeds 0–2 的相同 checkpoints。
- New：state-dependent Beta，`kappa(s) = 1.54 exp(c(s))`，concentration head 零初始化。
- Beta unit-space mean 使用 `sigmoid(2 mu)`，所以 bid/ask deterministic action 与 Gaussian 都严格为 `tanh(mu)`；mean-policy parameterization 不变。
- 两组其余 environment、observation、network 和 PPO 配置完全一致。

## Checkpoint comparison

Seed dispersion 的口径是：先计算每个 seed 的 two-side MAE，再对 3 个 per-seed MAE 取 population standard deviation。Worst-side 是 3 seeds × 2 sides 中最大的绝对 deterministic epsilon。

| Training step | Gaussian MAE | Beta MAE | Gaussian dispersion | Beta dispersion | Gaussian worst-side | Beta worst-side | Beta paired wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,480 | 0.0832 | 0.0780 | 0.0385 | 0.0104 | 0.1801 | 0.0965 | 1/3 |
| 60,416 | 0.1478 | 0.1623 | 0.0329 | 0.0413 | 0.3828 | 0.2468 | 1/3 |
| 100,352 | 0.1385 | 0.1395 | 0.0533 | 0.0608 | 0.2982 | 0.2552 | 2/3 |

Beta 在 20k checkpoint 有温和 aggregate 改善，但只改善 `1/3` seeds；60k 时 MAE 和 dispersion 均更差；100k 时 aggregate MAE 与 Gaussian 基本相同，但 dispersion 仍更差。没有形成随训练推进而逐渐拉开优势的轨迹。

## Final paired seeds

| Seed | Gaussian bid | Gaussian ask | Gaussian MAE | Beta bid | Beta ask | Beta MAE | Beta improved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.1161 | 0.2982 | 0.2071 | 0.0432 | 0.2552 | 0.1492 | Yes |
| 1 | -0.1155 | 0.0389 | 0.0772 | 0.1956 | 0.2219 | 0.2088 | No |
| 2 | 0.1176 | 0.1445 | 0.1310 | 0.0022 | 0.1191 | 0.0607 | Yes |

Beta 改善了 seeds 0 和 2，但 seed 1 明显变差。这解释了为何 paired-win 条件通过，而 aggregate MAE 与 seed stability 条件没有通过。

## Numerical and boundary sanity

- Distribution samples 均严格位于 action bounds 内；重复计算 sampled action 的 log probability finite 且一致。
- Arbitrary latent means 下，Beta transformed mean 与 Gaussian deterministic action 在 machine precision 内一致。
- 初始 Monte Carlo bid/ask action std：Beta `0.6270`，squashed Gaussian `0.6280`，初始 stochasticity 已合理匹配。
- 3 个 final stochastic evaluations 中没有 NaN/Inf。Bid/ask near-bound frequency 平均 `4.89%`，seed range 为 `2.92%–6.40%`；未观察到数值或边界 pathology。
- Beta sampled epsilon std 的三 seed 平均为 bid `0.6096`、ask `0.6065`，没有异常塌缩。

## Go / No-Go evaluation

| Pre-registered condition | Result |
| --- | --- |
| Aggregate MAE 明显降低 | **Fail**：`0.1385 → 0.1395` |
| Seed dispersion 不变差 | **Fail**：`0.0533 → 0.0608` |
| Worst-side 不恶化 | Pass：`0.2982 → 0.2552` |
| 至少 2/3 paired seeds 改善 | Pass：2/3 |
| 无 numerical/boundary pathology | Pass |

最终判断：直接定义在 bounded action space 上的 Beta geometry 在本次 screening 中没有清晰优于当前 state-dependent squashed Gaussian。结果差异来自 seed 间重新分配，而不是稳定、一致的 policy-learning 改善；因此保留现有 Gaussian candidate baseline，并停止 Beta 方向。

## Validation

- Existing unit tests：10/10 passed。
- Python compile：passed。
- Existing diagnostics：passed。
- Short Beta PPO smoke run：passed；common PPO diagnostics finite，动作严格在 bounds 内。
