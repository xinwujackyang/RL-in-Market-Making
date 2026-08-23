# GAE λ-Return Value Target 实验报告

## 结论

本实验属于 **Case B**：GAE λ-return 显著降低了记录到的 critic value loss 及其绝对峰值，但没有改善 long-run policy stability。相反，最终 deterministic two-side MAE、seed dispersion 和 worst-side error 都比 MC-target baseline 更差。

因此，critic target variance 是当前实现中一个真实且可显著降低的噪声来源，但它不是剩余 mean-policy drift 的主要解释。本结果保留为 PPO learning diagnostic；`gae_value_target` 继续默认关闭，不将 λ-return 组提升为 current best。按计划停止 critic-target investigation，下一步应检查 minibatch / advantage update semantics，不做 λ、TD、n-step 或 critic learning-rate sweep。

## Controlled comparison

两组唯一 conceptual difference 是 critic target：

- Baseline：actor 使用 normalized GAE advantage（λ=0.95），critic 拟合 full bootstrapped Monte Carlo return。
- New：actor 仍使用同一个 normalized GAE advantage，critic 改为拟合 `V_old(s_t) + raw GAE advantage_t`。

新组保持 five-feature relative-price observation、separate actor/critic、20 个 unit investors、Random `U[-1,1]` competitor、market sigma `0.2`、2×256 tanh、learning rate `5e-5`、clip `0.3`、gamma `0.999`、GAE λ `0.95`、10 epochs、minibatch 256、entropy coefficient `0.003`、value coefficient `0.5`、gradient clip `0.5`、bounded tanh-Gaussian、相同 5 seeds 和 204,800 steps。`kl_coef=0`。Baseline 直接复用 `results/relative_price/`，没有重跑。

## Final policy metrics

| 指标 | MC target baseline | GAE λ-return target | 变化 |
|---|---:|---:|---:|
| Final two-side MAE | 0.1601 | 0.2063 | +28.9%（更差） |
| Seed dispersion | 0.1424 | 0.1538 | +8.0%（更差） |
| Worst-side error | 0.4326 | 0.5785 | +33.7%（更差） |

这里的 seed dispersion 与 baseline 报告口径一致：先计算每个 seed 的 bid/ask 平均绝对误差，再对 5 个 per-seed MAE 取 population standard deviation。

新组 per-seed final deterministic 结果：

| Seed | Bid ε | Ask ε | Two-side MAE |
|---:|---:|---:|---:|
| 0 | 0.0675 | -0.0204 | 0.0440 |
| 1 | -0.2109 | -0.0184 | 0.1147 |
| 2 | 0.1983 | -0.0559 | 0.1271 |
| 3 | 0.3759 | 0.5785 | 0.4772 |
| 4 | -0.3017 | -0.2355 | 0.2686 |

相对 baseline，seeds 0 和 2 改善，seeds 1、3、4 变差。尤其 seed 3 保留 persistent bad-seed behavior，ask-side error 达到 `0.5785`；seed 4 也由 baseline 的低误差 seed 变为明显负 offset。因此不能把平均结果解释为更稳定。

## Value-loss behavior

以下统计来自两组相同的 8 个 checkpoint × 5 seeds，共 40 条 rollout-level `value_loss` 记录：

| Value-loss diagnostic | MC target baseline | GAE λ-return target | 变化 |
|---|---:|---:|---:|
| 全部 checkpoint median | 630.63 | 48.34 | -92.3% |
| 全部 checkpoint mean | 1,719.95 | 91.72 | -94.7% |
| 全部 checkpoint maximum | 18,014.06 | 521.65 | -97.1% |
| 100k+ median | 1,528.59 | 67.35 | -95.6% |
| 100k+ maximum | 18,014.06 | 521.65 | -97.1% |
| `value_loss > 100` 的记录数 | 39/40 | 8/40 | 明显减少 |

λ-return 的 absolute value-loss scale 和 large spikes 均大幅降低，尤其排除了 baseline 中 `8,177.6` 和 `18,014.1` 量级的 late critic-loss episodes。因为两组拟合的是不同 target，这些 MSE 数值不能视为完全同尺度的 critic accuracy comparison；但它们直接回答了本轮问题：匹配 GAE 的 λ-return 确实提供了明显低方差、较稳定的训练 target。

新组仍有 late value-loss 增长：五-seed mean 从 60k 的 `23.58` 增到 100k 的 `111.11`、150k 的 `131.85`、200k 的 `162.14`，最终 checkpoint 为 `182.57`，最大值为 `521.65`。所以 spikes 是大幅缩小，而不是完全消失。

## 100k+ policy trajectory

Policy drift 没有随 critic loss 的改善而消失：

- seed 3 从 100k 的 `(0.021, -0.152)` 漂到 200k 的 `(0.493, 0.544)`，204.8k 仍为 `(0.439, 0.585)`。
- seed 4 在 150k 漂到 `(0.275, 0.415)`，之后恢复并穿越到负 offset，204.8k 为 `(-0.134, -0.155)`，表现为明显 oscillation。
- seed 0 late recovery 较好，但 seed 1 保留负 bid offset，seed 2 仍有 moderate offset。

Checkpoint aggregate MAE 在 100k 时为 `0.1541`，但随后升到 150k 的 `0.1840`、200k 的 `0.1905`，204.8k 为 `0.1756`。因此 early/mid-run 的改善没有转化为可靠的 final seed stability。

## 对计划问题的直接回答

1. **Final MAE：`0.1601 → 0.2063`。** 没有改善，恶化约 28.9%。
2. **Seed dispersion：`0.1424 → 0.1538`。** 没有改善，恶化约 8.0%。
3. **Worst-side error：`0.4326 → 0.5785`。** catastrophic side 更严重。
4. **Value-loss spikes 是否明显减少？** 是。median、mean 和 maximum 均下降超过 90%，100k+ 极端峰值也大幅收缩。
5. **100k+ persistent drift 是否改善？** 否。seed 3 明显 runaway，seed 4 明显 oscillate；最终 policy 指标整体更差。

## Validation

- λ=1 sanity check：有限 rollout 加 bootstrap 时，`V_old + raw GAE advantage` 与 bootstrapped Monte Carlo return 数值对齐。
- Scale check：critic target 使用 raw advantage；`buffer.get()` 只 normalization actor advantage，不修改 target。
- 默认 `gae_value_target=False`，旧 MC-target 行为保留。
- 10/10 existing tests、Python compile、现有 diagnostics 和 short smoke run 通过。
- 正式输出包含 5 个 seeds、每 seed 204,800 steps，以及 8 个指定 checkpoint，共 40 条 trajectory rows。
