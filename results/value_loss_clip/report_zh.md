# Value-Loss Clipping 实验报告

## 结论

本实验属于 **Case D：stability 明显变差**。

在 MC critic target 不变的情况下，将每个 squared value error cap 在 `10.0` 并没有只移除少量 extreme gradients。Clipping 在全部 checkpoint 中平均作用于 `94.3%` 的 samples，100k+ 更达到 `97.6%`。Critic 因而几乎失去对绝大多数 samples 的 value gradient，unclipped prediction error 持续扩大，policy 在约 40k–60k 后形成更严重的 persistent runaway。

结果不支持“少量巨大 critic-error outliers 是当前 actor instability 主因”。在当前 value-error scale 下，`vf_clip_param=10` 是一个过于激进的 critic-learning intervention，而不是温和的 outlier guard。按计划不做 threshold sweep，不将 value clipping 纳入 current best，关闭这一方向。

## Controlled comparison

两组均使用 full bootstrapped Monte Carlo critic target。本轮唯一 conceptual change 是：

```text
Baseline value loss = mean(error²)
New value loss      = mean(min(error², 10))
```

新组保持 five-feature relative-price observation、separate actor/critic、joint global gradient clipping、20 个 unit investors、Random `U[-1,1]` competitor、market sigma `0.2`、2×256 tanh、learning rate `5e-5`、clip `0.3`、gamma `0.999`、GAE λ `0.95`、10 epochs、minibatch 256、entropy coefficient `0.003`、value coefficient `0.5`、gradient clip `0.5`、相同 5 seeds 和 204,800 steps。`gae_value_target=False`，`kl_coef=0`。Baseline 直接复用 `results/relative_price/`，没有重跑。

## Primary policy metrics

| 指标 | MC baseline | Value-loss clip 10 | 变化 |
|---|---:|---:|---:|
| Final two-side MAE | 0.1601 | 0.3640 | +127.3%（更差） |
| Seed dispersion | 0.1424 | 0.2170 | +52.3%（更差） |
| Worst-side error | 0.4326 | 0.8149 | +88.4%（更差） |

Seed dispersion 沿用 baseline 口径：先计算每个 seed 的 bid/ask 平均绝对误差，再对 5 个 per-seed MAE 取 population standard deviation。

新组 per-seed final deterministic 结果：

| Seed | Bid ε | Ask ε | Two-side MAE |
|---:|---:|---:|---:|
| 0 | 0.6159 | 0.8149 | 0.7154 |
| 1 | 0.3632 | 0.0939 | 0.2286 |
| 2 | 0.7154 | 0.3151 | 0.5153 |
| 3 | 0.3201 | 0.1206 | 0.2204 |
| 4 | -0.1289 | 0.1518 | 0.1403 |

Seeds 0 和 2 成为明显 catastrophic seeds；其余 seeds 也没有抵消 aggregate deterioration。

## Clipping activity

| Diagnostic | 全部 checkpoints | 100k+ checkpoints |
|---|---:|---:|
| Mean value clip fraction | 94.33% | 97.60% |
| Median value clip fraction | 96.15% | 98.49% |
| Minimum observed fraction | 66.88% | 91.92% |
| Maximum observed fraction | 99.90% | 99.90% |

这不是“1% samples 是 extreme outliers”的情形。即使最不活跃的 checkpoint，也有约三分之二 samples 超过 threshold；late training 中几乎所有 samples 都被 clipping。超过阈值后该 sample 对 critic value loss 的 gradient 为零，所以 critic update 被系统性削弱。

实际用于 optimization 的 clipped value loss 均值为 `9.6168`、median 为 `9.7491`、最大为 `9.9952`，长期贴近理论上限 `10`。因此 clipped loss 本身看起来平稳并不意味着 critic prediction 准确。

## Unclipped critic accuracy

| Unclipped value-loss diagnostic | Baseline | Value-loss clip 10 | 变化 |
|---|---:|---:|---:|
| 全部 checkpoint median | 630.63 | 4,429.04 | 7.0× |
| 全部 checkpoint mean | 1,719.95 | 53,591.88 | 31.2× |
| 全部 checkpoint maximum | 18,014.06 | 874,387.10 | 48.5× |
| 100k+ median | 1,528.59 | 17,188.78 | 11.2× |
| 100k+ mean | 2,653.66 | 104,884.44 | 39.5× |

Critic accuracy 明显恶化，而不是改善。`value_loss≈10` 只说明其 gradient influence 被 cap；`value_loss_unclipped` 表明 value prediction error 实际在 runaway。两者必须明确区分。

## Training trajectory

Aggregate checkpoint MAE：

| Step | Baseline | Value-loss clip 10 |
|---:|---:|---:|
| 20,480 | 0.1493 | 0.1668 |
| 60,416 | 0.1930 | 0.3834 |
| 100,352 | 0.1807 | 0.3727 |
| 149,504 | 0.1561 | 0.3200 |
| 199,680 | 0.1828 | 0.3643 |
| 204,800 | 0.1692 | 0.3550 |

Runaway 大约在 40k–60k 已经形成，并持续到训练结束：

- seed 0 在 100k 的 ask ε 已达 `0.769`，204.8k 仍为 `0.828`。
- seed 2 在 100k 的 bid ε 为 `0.755`，204.8k 仍为 `0.723`。
- seed 3 的 unclipped loss 在 199.7k 达到 `874,387`，204.8k 仍为 `678,905`；clipped loss 无法显示这一失真。

## 对计划问题的直接回答

1. **Final MAE 是否改善？** 否，`0.1601 → 0.3640`，恶化 127.3%。
2. **Seed dispersion 是否改善？** 否，`0.1424 → 0.2170`，恶化 52.3%。
3. **Worst-side / catastrophic drift 是否减少？** 否，worst-side `0.4326 → 0.8149`，并出现两个明显 bad seeds。
4. **有多少 critic samples 被 clipping？** 全程平均 94.3%，100k+ 平均 97.6%；clipping 几乎总是 active。
5. **Unclipped critic error 很大时 policy 是否更稳定？** 否。Unclipped error 比 baseline 大一个到两个数量级，policy 同时更不稳定。

## Decision

`vf_clip_param=10` 同时降低了 critic gradient influence 和 critic learning capacity；它没有带来 actor stability。由于本轮按预设 threshold 与 RLlib reference 对齐，且计划明确禁止 sweep，因此不尝试 `1/5/20/50/100`，不改 separate clipping、critic LR 或 optimizer。

下一步按计划转向 action-distribution parameterization，而不是继续 value-loss 方向。

## Validation

- Clamp check：`[1, 5, 20] → [1, 5, 10]`。
- Gradient check：低于 threshold 的 samples 保留正常 value gradient；明显高于 threshold 的 sample gradient 为零。
- 默认 `vf_clip_param=None`，旧实验 behavior 保留。
- 10/10 existing tests、Python compile、现有 diagnostics 和 short smoke run 通过。
- 正式输出包含 5 个 seeds、每 seed 204,800 steps，以及 8 个指定 checkpoints，共 40 条 trajectory rows。
