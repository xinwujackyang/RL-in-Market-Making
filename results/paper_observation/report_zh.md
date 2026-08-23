# Paper-Aligned Observation/Environment Baseline

## 结论

将 observation information set 与明确的 paper market volatility 一起对齐后，Random competitor replication gap **没有缩小，反而显著扩大**：

- Final bid MAE：`0.1512 -> 0.3896`；
- Final ask MAE：`0.1690 -> 0.3620`；
- Final two-side MAE：`0.1601 -> 0.3758`，增加约 134.7%；
- Per-seed MAE dispersion：`0.1424 -> 0.2406`，增加约 68.9%；
- Worst-side error：`0.4326 -> 0.8133`，增加约 88.0%；
- Latent sigma 仍约为 bid `0.984`、ask `0.986`，policy concentration 没有改善。

Seeds 0 和 1 出现明显 catastrophic policies，seed 4 也停留在约 `-0.41/-0.41`。只有 seeds 2 和 3 最终较接近 analytical optimum。因此：

> **这个 paper-aligned observation/environment setup 不应取代当前 five-feature relative-price baseline。更丰富的 paper-style market feedback 与 sigma=0.1 没有修复当前 custom PPO 的 seed-dependent long-run instability。按照预定决策，本项目应停止继续手工搜索 observation features，下一步转向 RLlib / framework-level reference comparison。**

本实验同时改变了 observation information set 和 market volatility，不能把恶化因果归于某一个 feature 或 `sigma=0.1`。

## Setup 定义

Paper-aligned mode 使用 24 维 observation：

```text
[previous own trade for investor 1, ..., investor 20,
 inventory,
 relative price,
 previous market share,
 previous inventory PnL]
```

每个 previous trade 为：

- `+1`：上一 timestep RL dealer 买入该 investor order；
- `-1`：上一 timestep RL dealer 卖出；
- `0`：该 order 被 competitor 赢走。

Previous market share 定义为上一 timestep RL fill count / 20。所有 previous-trade entries 和 market share 在 reset 时为 0，并在 routing/PnL 完成后写入 next observation，未改变 event timing。

实验运行同一 5 seeds、每 seed 204,800 steps；baseline 直接复用已有 relative-price 结果，没有重跑。

## 已经对齐的部分

```text
20 unit-size investors
50/50 buy/sell directions
mu = 0
sigma = 0.1
previous per-investor own trades
inventory
reference price information
previous market-share feedback
previous inventory PnL
```

## 有意保留的 numerical corrections

```text
relative price instead of raw price
separate actor / critic networks
corrected bounded tanh-Gaussian PPO
corrected routing and dealer-side signs
corrected event timing and PnL accounting
```

Relative price 有已有 conditioning evidence：raw price 曾导致约 97% 的第一层 tanh saturation。Separate actor/critic 与 corrected bounded actions 同样基于前序 correctness/stability 结果保留，而不是恢复旧的 clipping execution。

## 尚未对齐的部分

Current simulator does not reproduce the paper's stochastic reference bid/ask spread-curve process, so this component is not claimed to be exactly replicated.

当前 unit-size simulator 的 reference spread 基本是 current price 的确定函数，因此本轮没有为了形式一致加入 redundant spread scalar。本实验应称为 **paper-aligned observation/environment baseline**，不是 exact paper reproduction。

## 1. Analytical best-response accuracy

Random competitor 的 benchmark 为 `epsilon*=0`。

| Final metric | Relative-price reference | Paper-aligned setup | Change |
|---|---:|---:|---:|
| Bid MAE | 0.1512 | 0.3896 | +157.6% |
| Ask MAE | 0.1690 | 0.3620 | +114.2% |
| Two-side MAE | 0.1601 | 0.3758 | +134.7% |

Final policies：

| Seed | det bid | det ask | Two-side MAE | Worst-side error |
|---:|---:|---:|---:|---:|
| 0 | 0.5074 | -0.7981 | 0.6527 | 0.7981 |
| 1 | -0.8133 | -0.4191 | 0.6162 | 0.8133 |
| 2 | 0.0357 | 0.1408 | 0.0883 | 0.1408 |
| 3 | -0.1810 | -0.0381 | 0.1095 | 0.1810 |
| 4 | -0.4104 | -0.4139 | 0.4121 | 0.4139 |

Mean-policy accuracy 明显恶化，且 side-specific offsets 在不同 seeds 间方向不一致，不是单一系统 bias。

## 2. Seed stability

Per-seed two-side MAE dispersion 从 `0.1424` 增至 `0.2406`。Worst-side error 从 `0.4326` 增至 `0.8133`，并出现两个 `|epsilon|` 约为 0.8 的 catastrophic seeds。

因此 richer market feedback 没有提供稳定的 cross-seed learning signal；它至少在当前 corrected custom PPO 下引入或暴露了更强的 path dependence。

## 3. Long-run trajectory

| Step | Reference MAE | Paper-aligned MAE | Reference dispersion | Paper-aligned dispersion |
|---:|---:|---:|---:|---:|
| 20,480 | 0.1493 | 0.1319 | 0.1598 | 0.0676 |
| 60,416 | 0.1930 | 0.3465 | 0.1780 | 0.1493 |
| 100,352 | 0.1807 | 0.3912 | 0.1979 | 0.2000 |
| 149,504 | 0.1561 | 0.2588 | 0.0722 | 0.1330 |
| 199,680 | 0.1828 | 0.3034 | 0.1330 | 0.1925 |
| 204,800 | 0.1692 | 0.3646 | 0.1491 | 0.2050 |

20k 时新组表现尚可，但 20k-60k 后迅速 drift：

- seed 0 从 20k 的约 `(-0.14,-0.19)` 演化到 204.8k 的 `(0.48,-0.75)`，同时出现方向反转和持续 ask offset；
- seed 1 从 20k 的 `(0.09,-0.02)` drift 到 60k 的 `(-0.73,-0.40)`，此后 bid 长期停留在大幅负区间；
- seed 4 从 20k 起逐渐形成双侧约 `-0.4` 的 persistent offset；
- seeds 2 和 3 后期回到较接近 0 的区域，说明 instability 仍高度 seed-dependent。

因此 persistent drift / oscillation 不仅仍存在，而且相较 reference baseline 更严重。

## 4. Policy variance

Final latent sigma：

| Metric | Mean +/- seed std |
|---|---:|
| Latent sigma bid | 0.9837 +/- 0.0118 |
| Latent sigma ask | 0.9861 +/- 0.0056 |
| Sampled epsilon std bid | 0.5940 +/- 0.0476 |
| Sampled epsilon std ask | 0.6070 +/- 0.0765 |

Latent sigma 仍接近初始值 1。Paper-aligned information set 和较低 market volatility 没有改善 policy-variance learning；但按照本轮边界，不进一步调查 variance mechanism。

## Secondary metrics

Market share 为 `0.5847 +/- 0.0820`，spread PnL / step 为 `0.1163 +/- 0.0103`。这些 secondary metrics 不改变 analytical accuracy 和 seed stability 结论，也未据此展开 PnL investigation。

## 最终决策

1. **Mean accuracy：没有改善，two-side MAE 达到 0.3758。**
2. **Seed stability：明显恶化，dispersion 达到 0.2406。**
3. **Trajectory：60k 后出现多个 persistent/catastrophic drifts。**
4. **Policy variance：latent sigma 仍约为 1。**

由于 mean 远高于 `0.15-0.20` 且仍有 catastrophic seeds，按预先设定的决策规则：

> 不再继续手工搜索 observation features；下一步进入 RLlib / framework-level comparison。

## Validation

- Paper observation dimension：24；
- Reset：20 个 previous trades 与 market share 均为 0；
- Routing timing：next observation 正确包含上一 timestep signed own trades 与 market share；
- Python compile：通过；
- Existing tests：10/10 通过；
- Existing diagnostics：通过；
- Short smoke run：通过。
