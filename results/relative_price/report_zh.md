# Relative Price Scaling Experiment

## 结论

把 observation 中唯一的 raw price `P_t` 改成 `P_t / P_0 - 1` 后，actor 第一层 saturation 从 raw-price diagnostic 的约 `97%` 大幅降至约 `33%-34%`。Final deterministic two-side MAE 从 `0.3582` 降至 `0.1601`，worst-side error 从 `0.9158` 降至 `0.4326`，不再出现 `|epsilon|` 约为 `0.9` 的 catastrophic seed。

但改善并不等于稳定收敛：per-seed MAE dispersion 只从 `0.1608` 降至 `0.1424`，下降约 11.5%；100k steps 后 seed 3 仍持续向正方向 drift，seed 0 则出现大幅反向回归。因此本轮支持：

> **Raw price scaling 是 actor saturation 和平均 long-run error 的重要来源，但不是剩余 replication gap 的唯一原因。Relative price 消除了大部分 saturation 和最极端 seed，却没有消除 seed-specific drift/oscillation。**

按照停止条件，本轮不继续 scale inventory/PnL，也不做 normalization 或 PPO 调参；若继续调查，应转向 observation alignment。

## 实验控制

唯一主动变化：

```text
raw:      price_feature = P_t
relative: price_feature = P_t / P_0 - 1
```

其余配置保持不变：20 个 unit investors、Random competitor `U[-1,1]`、market sigma `0.2`、separate actor/critic `2 x 256 tanh`、learning rate `5e-5`、PPO clip `0.3`、minibatch `256`、10 epochs、entropy coefficient `0.003`、相同 reward/PPO/bounded tanh-Gaussian actions、相同 5 seeds 和 204,800 training steps。Raw-price baseline 直接读取已有结果，没有重跑。

验证确认 relative-price mode 在 inception 时满足 `P_0 / P_0 - 1 = 0`。现有 unit tests、Python compile、smoke run 和 actor/critic gradient-isolation diagnostic 均通过。

## 1. Saturation 从约 97% 降到了多少？

Saturation 定义为 deterministic evaluation states 上所有第一层 hidden units 的：

```text
P(|tanh(W_1 s + b_1)| > 0.95)
```

| Checkpoint | Raw price, seed 1 | Relative price, seed 1 | Relative price, 5-seed mean +/- std |
|---:|---:|---:|---:|
| 20,480 | 96.92% | 37.14% | 33.28% +/- 7.40 pp |
| 60,416 | 97.16% | 39.77% | 34.44% +/- 4.75 pp |

Raw baseline diagnostic 只运行过 seed 1，因此最严格的 paired comparison 是 seed 1：saturation 分别下降 `59.78` 和 `57.40` percentage points。Relative-price 五个 seeds 的 saturation 均在约 `25%-46%`，说明下降不是单个 seed 的偶然结果。

结论：**raw price 确实是第一层 saturation 的主要来源之一。** Relative transform 没有把 saturation 降到 0，但已消除约三分之二的 saturated activations。

## 2. Final two-side MAE：0.3582 -> 0.1601

Final metrics 使用与 raw baseline 相同的独立 deterministic evaluation streams。

| Metric | Raw price | Relative price | Change |
|---|---:|---:|---:|
| Bid MAE | 0.4413 | 0.1512 | -65.7% |
| Ask MAE | 0.2751 | 0.1690 | -38.6% |
| Two-side MAE | 0.3582 | 0.1601 | -55.3% |

Relative price 对原本较差的 bid side 改善尤其明显。平均 policy 更接近 Random competitor 的 analytical optimum `epsilon*=0`。

Final relative-price policies：

| Seed | det bid | det ask | Two-side MAE | Worst-side error |
|---:|---:|---:|---:|---:|
| 0 | -0.0545 | -0.1550 | 0.1048 | 0.1550 |
| 1 | -0.1005 | 0.0257 | 0.0631 | 0.1005 |
| 2 | 0.1667 | 0.1787 | 0.1727 | 0.1787 |
| 3 | 0.4326 | 0.4249 | 0.4287 | 0.4326 |
| 4 | -0.0018 | -0.0606 | 0.0312 | 0.0606 |

## 3. Seed dispersion：0.1608 -> 0.1424

Per-seed two-side MAE dispersion 从 `0.1608` 降至 `0.1424`，下降约 11.5%。这是改善，但幅度明显小于 mean MAE 的 55.3% 降幅。

因此 seed stability 的判断是：

- typical seed 明显更接近 analytical optimum；
- catastrophic tail 明显改善；
- 但 seed 3 仍显著偏离其他 seeds，所以 across-seed dispersion 只得到 modest improvement。

## 4. Worst-side error 是否改善？

**明显改善。** Worst-side error 从 raw baseline 的 `0.9158` 降至 `0.4326`，下降约 52.8%。Relative-price 组没有出现类似 raw seed 1 `bid=-0.916` 的 `|epsilon|` 约为 `0.9` catastrophic policy。

不过 seed 3 的 `bid=0.433, ask=0.425` 仍是明确的 bad seed；更准确的结论是 extreme failure 被削弱，而不是 seed dependence 被消除。

## 5. 100k 之后是否仍有 drift / oscillation？

**仍有明显的 seed-specific drift/oscillation，但 aggregate runaway 明显减轻。**

| Step | 5-seed trajectory MAE | Seed dispersion | Worst-side error |
|---:|---:|---:|---:|
| 100,352 | 0.1807 | 0.1979 | 0.6589 |
| 149,504 | 0.1561 | 0.0722 | 0.3743 |
| 199,680 | 0.1828 | 0.1330 | 0.4796 |
| 204,800 | 0.1692 | 0.1491 | 0.4884 |

Aggregate MAE 在 100k 后维持约 `0.16-0.18`，没有出现 raw-price baseline 那种整体 runaway。但个别路径仍不稳定：

- seed 3 的 trajectory MAE 从 100k 的 `0.0632` 增至 200k 附近的 `0.44-0.46`；
- seed 0 从 100k 的 `0.5755` 回落至 final `0.0924`，显示明显 oscillation/late recovery；
- seeds 1、2、4 相对稳定，final MAE 分别约为 `0.063`、`0.173`、`0.031`。

因此 relative scaling 显著改善整体 level，但不能解释或修复全部 long-run path dependence。

## Policy variance 补充

Final latent sigma 仍接近 1：bid `0.9797 +/- 0.0082`，ask `0.9737 +/- 0.0103`。这说明 observation scaling 改善的是 mean-policy representation/optimization，而不是 latent variance learning。不要把 mean MAE 的改善解释为 policy distribution 已明显收缩。

## 最终判断

1. **Saturation：**约 `97% -> 33%-34%`，大幅下降。
2. **Final two-side MAE：**`0.3582 -> 0.1601`，下降 55.3%。
3. **MAE seed dispersion：**`0.1608 -> 0.1424`，仅下降 11.5%。
4. **Worst-side error：**`0.9158 -> 0.4326`，catastrophic seed 明显缓解。
5. **100k 后 drift：**aggregate runaway 减轻，但 seed-specific drift/oscillation 仍存在。

综合而言，raw price scaling 是真实且重要的问题；修正后剩余不稳定性更可能与 observation content/alignment 有关，而不是继续 scale 其他 feature 就能自然消失。
