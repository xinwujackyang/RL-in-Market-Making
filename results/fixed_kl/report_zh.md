# Fixed Analytic KL Penalty Experiment

## 结论

在 current custom PPO clipping loss 上增加 fixed analytic KL penalty `beta=0.2`，得到的是**部分 tail stabilization，但整体 mean-policy accuracy 变差**：

- Final two-side MAE：`0.1601 -> 0.1988`，恶化 24.2%；
- Per-seed MAE dispersion：`0.1424 -> 0.1033`，改善 27.5%；
- Worst-side error：`0.4326 -> 0.4088`，仅改善 5.5%；
- 20k 后没有出现向 `0.8` 附近持续 runaway 的 seed，但多数 seeds 逐渐形成约 `+0.15` 至 `+0.40` 的正向 mean offset；
- Fixed-KL sampled approx-KL 相对 no-KL 只下降约 12.7%，analytic KL 也出现过 `0.0832` 的单-update spike；
- Latent sigma 仍约为 `0.98-0.99`。

因此本轮不是干净的 Case A：seed dispersion 确实下降，但主要代价是多个原本较准确的 seeds 变成较一致的 moderate error。它也不是典型 Case D，因为 policy 没有被锁在 initialization 附近，而是仍发生了累计移动。

更准确的判断是：

> **Fixed beta=0.2 对 catastrophic tail 有有限约束，但没有实质降低 per-update policy KL，也没有改善 analytical accuracy 或 long-run trajectory。Simple fixed KL 不能解释 RLlib 的稳定性优势；本轮停止 KL 方向，不自动做 beta sweep 或 adaptive KL。下一步应优先检查 advantage/minibatch/update semantics。**

## 实验控制

唯一算法变化：

```text
loss = clipped_policy_loss
     + 0.5 * value_loss
     - 0.003 * entropy
     + 0.2 * KL(old latent Gaussian || current latent Gaussian)
```

每次 rollout 完成后、任何 optimizer step 之前，一次性计算并冻结所有 rollout observations 的 old latent means，以及 old global log-std。整个 10 epochs × minibatches 期间都与这一个 rollout policy 比较。

KL 使用三维 diagonal Gaussian 的 exact distribution formula，先逐 action dimension 计算，再求和并对 minibatch 取均值。现有 `old_log_prob - new_log_prob` 只保留为 sampled `approx_kl` diagnostic，没有用于 KL loss。

Environment、five-feature relative-price observation、20 unit investors、Random competitor、sigma `0.2`、separate actor/critic、network、reward、advantages、bounded tanh-Gaussian、learning rate、clip、epochs、minibatch 和相同 5 seeds 均保持不变。No-KL baseline 没有重跑。

## 1. Fixed KL 是否降低 final MAE？

**没有。Final two-side MAE 增加 24.2%。**

| Final metric | No KL | Fixed KL | Change |
|---|---:|---:|---:|
| Bid MAE | 0.1512 | 0.2123 | +40.4% |
| Ask MAE | 0.1690 | 0.1853 | +9.6% |
| Two-side MAE | 0.1601 | 0.1988 | +24.2% |

Final fixed-KL policies：

| Seed | det bid | det ask | Two-side MAE | Worst-side error |
|---:|---:|---:|---:|---:|
| 0 | 0.3319 | 0.2493 | 0.2906 | 0.3319 |
| 1 | 0.0621 | 0.2595 | 0.1608 | 0.2595 |
| 2 | 0.4088 | 0.2553 | 0.3321 | 0.4088 |
| 3 | 0.1815 | 0.1575 | 0.1695 | 0.1815 |
| 4 | 0.0774 | -0.0048 | 0.0411 | 0.0774 |

Paired comparison 中，baseline bad seed 3 从 MAE `0.4287` 改善到 `0.1695`；但 seeds 0、1、2 分别从 `0.1048/0.0631/0.1727` 恶化到 `0.2906/0.1608/0.3321`。因此 fixed KL 不是普遍提高 accuracy，而是改变了 seed error 分布。

## 2. Seed dispersion 是否明显下降？

**有中等幅度下降，但含义不是全面稳定收敛。** Dispersion 从 `0.1424` 降至 `0.1033`，下降 27.5%。

Fixed KL 消除了一个特别差、其余 seeds 较好的 error pattern，但新组四个 seeds 都形成不同程度的正偏移。更低 dispersion 部分来自 errors 更均匀，而不是所有 seeds 都更接近 0。

## 3. Persistent catastrophic drift 是否减少？

**Extreme runaway 有所减少，但 aggregate long-run drift 没有改善。**

| Step | No-KL MAE | Fixed-KL MAE | No-KL dispersion | Fixed-KL dispersion | Fixed worst-side |
|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.1493 | 0.1241 | 0.1598 | 0.0876 | 0.3703 |
| 60,416 | 0.1930 | 0.2451 | 0.1780 | 0.0985 | 0.4709 |
| 100,352 | 0.1807 | 0.2007 | 0.1979 | 0.1305 | 0.4552 |
| 149,504 | 0.1561 | 0.2201 | 0.0722 | 0.1132 | 0.4710 |
| 199,680 | 0.1828 | 0.2162 | 0.1330 | 0.0859 | 0.3783 |
| 204,800 | 0.1692 | 0.2070 | 0.1491 | 0.1046 | 0.3877 |

20k 时 fixed KL 较好，但 40k-60k 后 five-seed MAE 一直高于 no-KL reference。Worst side 大致限制在 `0.47` 内，没有出现 `0.6-0.8` 的 trajectory runaway；代价是 policy 在较宽的 moderate-offset 区间长期停留。

所以 fixed KL 改善的是 extreme movement 的上尾，不是向 analytical optimum 的 convergence。

## 4. Analytic KL 是否被实质性压低？

**没有明确证据表明被实质性压低。** 因为历史 no-KL run 没有记录 analytic distribution KL，不能做严格 analytic-to-analytic comparison；计划要求 baseline 不重跑，因此这里使用同口径 sampled approx-KL 作为有限对照。

| Update diagnostic | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Fixed-KL analytic KL | 0.00638 | 0.01255 | 0.00014 | 0.08320 |
| Fixed-KL sampled approx-KL | 0.00571 | 0.01238 | -0.00105 | 0.07992 |
| No-KL sampled approx-KL | 0.00654 | 0.00580 | -0.00137 | 0.02922 |

Fixed-KL sampled approx-KL 只比 no-KL 低约 12.7%，同时 max spike 更大。`beta=0.2` 的梯度确实进入 loss，但没有构成严格 trust region 或可靠地阻止单个 rollout update 的大 KL。

Checkpoint aggregate 中，analytic KL 与 sampled estimator 也不完全一致。例如 20k 为 `0.00663` 对 `0.00384`，100k 为 `0.00413` 对 `0.00656`，204.8k 为 `0.00234` 对 `0.00339`。Sampled approx-KL 可以为轻微负值，而 exact analytic KL 在全部记录中保持非负。

最大的 analytic KL spike 来自 seed 4 在 149k 的 `0.0832`，但该 seed 的 mean policy 最终最好。这也说明单次 rollout-update KL 大小与累计 long-run mean offset 不是简单一一对应。

## 5. Latent sigma 不变时，mean-policy stability 是否改善？

Final latent sigma 基本不变：

| Metric | Mean +/- seed std |
|---|---:|
| Latent sigma bid | 0.9833 +/- 0.0081 |
| Latent sigma ask | 0.9824 +/- 0.0091 |
| Sampled epsilon std bid | 0.6910 +/- 0.0680 |
| Sampled epsilon std ask | 0.7154 +/- 0.0551 |

在 variance 没有 collapse 的情况下，fixed KL 确实降低了 MAE dispersion 和 extreme tail；但 final mean MAE 与大部分 paired seeds 都变差。因此只能称为有限的 distributional stabilization，不能称为 mean-policy learning 改善。

## 最终判断

1. **Final MAE：没有降低，反而恶化 24.2%。**
2. **Seed dispersion：下降 27.5%，但部分来自 errors 更均匀。**
3. **Catastrophic drift：extreme tail 减少，moderate persistent offsets 增多。**
4. **Analytic KL：loss/gradient 实现有效，但 update KL 没有被实质压低。**
5. **Latent sigma：仍约 0.98，mean accuracy 没有改善。**

综合结果不足以支持下一步自动进入 fixed-vs-adaptive KL。按照本轮边界，停止 KL 方向；若继续 learner diagnosis，优先比较 advantage/minibatch/update semantics。

## Validation

- KL identity：相同 distribution 得到 `0.0`；
- Perturbed distribution：KL=`0.02223 > 0`；
- Gradient：analytic KL backward 后 actor hidden、policy mean、`log_std` 均得到非零 gradient；
- 所有记录的 analytic KL 均非负；
- Existing tests：10/10 通过；
- Existing diagnostics：通过；
- Python compile：通过；
- Short fixed-KL smoke run：通过。
