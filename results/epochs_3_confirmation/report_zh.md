# Reduced PPO batch reuse long-run confirmation

## 结论

`n_epochs=3` **不能升级为新的正式 baseline**。

在 60k–100k steps，3-epoch policy 明显优于现有 10-epoch baseline；但该优势没有维持到完整训练预算。相同 204,800-step final evaluation 下：

- Two-side MAE：`0.0808 → 0.1021`，恶化 `26.4%`。
- Seed dispersion：`0.0467 → 0.0376`，下降 `19.5%`。
- Worst-side error：`0.2250 → 0.1977`，下降 `12.2%`。
- Paired-seed wins：仅 `1/5`。

虽然 dispersion 和 worst-side 有改善，但 primary MAE 变差且多数 paired seeds 恶化，不满足预先规定的升级条件。最终结论是：

> reduced batch reuse helps early/mid training, but does not robustly solve long-run instability.

## Controlled setup

新组只将 `n_epochs` 从 `10` 改为 `3`。State-dependent squashed Gaussian、`ent_coef=0.003`、environment、observation、network 和其余 PPO 配置全部保持 current baseline。运行规格为 5 seeds、200 rollouts、horizon 1024，共 204,800 training steps。

10-epoch baseline 没有重跑，直接复用 `results/state_dependent_std/` 中已有的 5-seed / 204,800-step results。Final metrics 使用两组相同 evaluation streams；long-run trajectory 使用相同 rollout-aligned checkpoints。

## Final 204,800-step comparison

| Metric | 10 epochs | 3 epochs | Change |
| --- | ---: | ---: | ---: |
| Two-side MAE | 0.0808 | 0.1021 | +26.4% |
| MAE seed dispersion | 0.0467 | 0.0376 | -19.5% |
| Worst-side error | 0.2250 | 0.1977 | -12.2% |
| Paired-seed wins | — | 1/5 | — |

### Per-seed final policy

| Seed | 10-epoch bid | 10-epoch ask | 10-epoch MAE | 3-epoch bid | 3-epoch ask | 3-epoch MAE | 3 epochs improved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.0083 | -0.0412 | 0.0247 | 0.0447 | 0.0790 | 0.0618 | No |
| 1 | -0.0215 | 0.0432 | 0.0324 | 0.0779 | 0.1089 | 0.0934 | No |
| 2 | 0.2250 | -0.0295 | 0.1272 | 0.1977 | -0.0960 | 0.1469 | No |
| 3 | 0.1677 | 0.1078 | 0.1377 | -0.0518 | 0.0750 | 0.0634 | Yes |
| 4 | 0.0362 | 0.1276 | 0.0819 | 0.1310 | -0.1593 | 0.1452 | No |

3-epoch 组没有出现单个 `|epsilon|` 特别巨大的 catastrophic seed，但 seeds 0、1、2、4 的 two-side MAE 均高于 paired baseline。更窄的 seed distribution 因而不是来自整体 policy quality 提升，而是来自多个 seeds 聚集在中等误差区间。

## Long-run trajectory

| Step | 10-epoch MAE | 3-epoch MAE | 10-epoch dispersion | 3-epoch dispersion | 10-epoch worst-side | 3-epoch worst-side | Paired wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,480 | 0.0759 | 0.0707 | 0.0444 | 0.0459 | 0.1801 | 0.1524 | 2/5 |
| 60,416 | 0.1153 | 0.0680 | 0.0521 | 0.0278 | 0.3828 | 0.1925 | 5/5 |
| 100,352 | 0.1309 | 0.0611 | 0.0495 | 0.0287 | 0.2982 | 0.2002 | 5/5 |
| 149,504 | 0.1151 | 0.0861 | 0.0323 | 0.0357 | 0.2309 | 0.2273 | 3/5 |
| 199,680 | 0.0621 | 0.0923 | 0.0188 | 0.0505 | 0.1930 | 0.1905 | 2/5 |
| 204,800 | 0.0695 | 0.0939 | 0.0337 | 0.0442 | 0.2110 | 0.2093 | 1/5 |

Trajectory 清楚显示三个阶段：

1. **60k–100k：** 3 epochs 的 MAE 约为 baseline 的一半，5/5 seeds 改善，screening 结果得到复现。
2. **149.5k：** 优势开始缩小，dispersion 已略高于 baseline，paired wins 降到 3/5。
3. **199.7k–204.8k：** MAE 优势反转，dispersion 明显更高，paired wins 分别降至 2/5 和 1/5。

因此 100k 后确实重新出现 drift / offset；较少 batch reuse 延迟或减轻了 early/mid instability，但没有在 long run 中稳定收敛到更好的 mean policy。

## Approx KL 与 clip fraction

| Step | 10-epoch KL | 3-epoch KL | 10-epoch clip fraction | 3-epoch clip fraction |
| ---: | ---: | ---: | ---: | ---: |
| 20,480 | 0.00729 | 0.00443 | 0.0321 | 0.0114 |
| 60,416 | 0.00501 | 0.00367 | 0.0187 | 0.0128 |
| 100,352 | 0.00540 | 0.00070 | 0.0194 | 0.0000 |
| 149,504 | 0.00758 | 0.00032 | 0.0270 | 0.0069 |
| 199,680 | 0.00406 | 0.00323 | 0.0258 | 0.0147 |
| 204,800 | 0.01139 | 0.00087 | 0.0676 | 0.0058 |

3-epoch 组在所有 checkpoints 都有更低的 aggregate approx KL 和 clip fraction；204.8k 时两者分别低约 `92.4%` 和 `91.4%`。但此时 policy MAE 已更差。这说明减少 batch reuse 确实限制了单次 rollout 的 policy movement，却不足以保证 long-run mean-policy convergence；KL / clipping 变小本身不是成功标准。

## Economic sanity check

| Metric | 10 epochs | 3 epochs |
| --- | ---: | ---: |
| Market share | 0.4804 | 0.5003 |
| Spread PnL / step | 0.1174 | 0.1178 |
| Total PnL / step | 0.1047 | 0.0625 |
| Inventory std | 9.1177 | 8.5228 |

Market share、spread capture 和 inventory dispersion 没有明显异常。Total PnL / step 更低，但本轮不围绕 noisy PnL 展开额外 investigation。

## Decision

不将 `n_epochs=3` 升级为新 baseline，继续保留 `n_epochs=10` 作为 current reference。Reduced batch reuse 方向到此停止；本轮不做 epoch sweep、learning-rate compensation 或其他 PPO mechanism 修改。

## Validation

- Existing unit tests：10/10 passed。
- Python compile：passed。
- Short `n_epochs=3` PPO smoke：passed。
- 5 seeds 全部完成；每个 seed 恰好包含 6 个指定 checkpoints。
- CSV 无 NaN/Inf，配置均为 `n_epochs=3`、`ent_coef=0.003`、204,800 steps。
