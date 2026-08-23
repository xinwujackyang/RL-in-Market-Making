# Input Scaling / Actor Saturation Diagnostic

## 结论

使用 separate actor/critic 实验中较早 drift 的 **seed 1**，在 20,480 与 60,416 steps 对相同的 3 episodes x 5 days deterministic evaluation paths 做诊断。

结果是：

- Observation distribution 从 20k 到 60k **没有膨胀**；inventory 与 PnL dispersion 反而略降。
- Actor 第一层在两个 checkpoint 都已处于**极端 saturation**：`96.92% -> 97.16%` 的 pre-activations 满足 `|tanh(u)|>0.95`。
- Saturation 没有随 drift 明显增强，但其绝对水平本身已构成明确的 numerical-health 问题。

因此本轮归类为：

> **Case A 的静态重度 saturation 情形：input scaling 值得继续做一次受控 fixed-scaling 验证，但当前证据不能证明 20k 到 60k 的 drift 是由 observation distribution deterioration 触发。**

本轮只诊断，没有实施 normalization、clipping、LayerNorm 或 observation redesign。

## Seed 选择

选择 seed 1，因为上一轮 separate trajectory 中：

| Step | deterministic bid | deterministic ask |
|---:|---:|---:|
| 20,480 | -0.013 | -0.467 |
| 39,936 | 0.026 | -0.609 |
| 60,416 | -0.442 | -0.567 |

它在 40k-60k 已出现明确 mean-policy drift，符合计划的单-seed选择标准。

## Q1：Observation scale 是否存在明显异常或膨胀？

不同 features 的 raw units 差异很大：price 约为 100，inventory std 约为 7，PnL std 约为 2，而 hedge cost 约为 0.1。这种 unit mismatch 只作为背景，本身不自动证明需要 normalization。

| Feature | 20k mean / std | 20k p01 / p50 / p99 | 60k mean / std | 60k p01 / p50 / p99 |
|---|---:|---:|---:|---:|
| inventory | -3.499 / 7.304 | -17.399 / -3.633 / 12.983 | -0.442 / 7.250 | -15.512 / -0.692 / 15.242 |
| price | 99.882 / 1.608 | 96.681 / 99.784 / 103.494 | 99.882 / 1.608 | 96.681 / 99.784 / 103.494 |
| total PnL | 0.031 / 2.043 | -6.169 / 0.096 / 5.948 | 0.035 / 1.868 | -5.367 / 0.079 / 5.060 |
| inventory PnL | -0.072 / 2.044 | -6.354 / 0.000 / 5.834 | -0.039 / 1.864 | -5.456 / -0.016 / 5.007 |
| hedge cost | 0.091 / 0.069 | 0.000 / 0.079 / 0.304 | 0.085 / 0.067 | 0.000 / 0.073 / 0.296 |

关键 features 没有 distribution expansion：

- inventory std：`7.304 -> 7.250`；
- total-PnL std：`2.043 -> 1.868`；
- inventory-PnL std：`2.044 -> 1.864`；
- PnL p01/p99 tails 均收窄；
- price statistics 完全相同，因为 checkpoints 使用相同的外生 evaluation paths。

所以没有证据支持 `policy drift -> state distribution expansion -> instability` 这条具体链路。

## Q2：Actor 第一层是否明显 saturation？

**是，绝对 saturation 极其严重，但没有随 drift 显著恶化。**

| Step | E[abs(u)] | P(abs(tanh(u)) > 0.95) |
|---:|---:|---:|
| 20,480 | 23.489 | 96.920% |
| 60,416 | 23.520 | 97.162% |
| Change | +0.030 (+0.13%) | +0.242 percentage points |

约 97% 的 observation-hidden-unit pairs 位于第一层 tanh saturation 区，远高于“低 saturation”的健康情形。Raw price 约 100，而其他 features 小几个数量级，与这一现象在数值上相容。

但是 20k 到 60k 的变化很小：mean absolute pre-activation 只增加 0.13%，saturation fraction 只增加 0.24 percentage points。因此该诊断证明的是**持续存在的严重 saturation**，不是 drift 开始时新出现的 saturation deterioration。

## Q3：Scaling 是否值得继续调查？

**值得做一次最小、受控的 fixed input-scaling experiment。** 理由不是 observation tails 在 60k 膨胀，而是 actor 第一层从 20k 起已有约 97% saturation，这提供了比单纯 units mismatch 更直接的 neural-network evidence。

结论边界：

- 支持：当前 raw input representation 使 actor 第一层 numerically unhealthy；
- 不支持：20k 到 60k 的 distribution expansion 导致 drift；
- 尚未证明：降低第一层 saturation 会消除 mean-policy drift。

下一轮如果继续，应只测试预先固定的简单 feature scaling，并保持其他配置完全不变。本轮按停止条件没有自动实施任何 scaling。

