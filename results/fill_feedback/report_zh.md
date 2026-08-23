# Previous Fill Feedback Incremental Observation Experiment

## 结论

在 relative-price observation 后增加上一 timestep 的 bid-side / ask-side fill fractions，**没有进一步改善最终 mean-policy accuracy 或实质性改善 seed stability**：

- Final deterministic two-side MAE：`0.1601 -> 0.2124`，恶化 32.7%；
- Per-seed MAE dispersion：`0.1424 -> 0.1386`，仅下降 2.7%；
- Worst-side error：`0.4326 -> 0.4822`，恶化 11.5%；
- 40k-100k 的 trajectory 一度更接近 0，但 100k 后 seeds 0 和 2 出现新的持续 drift。

因此本轮支持：

> **Compressed previous-fill feedback 不能解释当前剩余 replication gap。它改善了部分 early/mid-training trajectory，但没有转化为更好的 long-run final policy；bad-seed failure 主要是从 baseline seed 3 迁移到了 fill-feedback seeds 0 和 2。**

按照停止条件，本轮不继续增加 market share、volume 或逐 investor identity features。下一步若继续，应单独测试删除非 paper-aligned 的 `TotalPnL` / `HedgeCost` features。

## 实验控制与 feature 定义

唯一主动变化是在现有 5 维 relative-price observation 后追加：

```text
prev_bid_fill_fraction = previous RL bid-side fill count / 20
prev_ask_fill_fraction = previous RL ask-side fill count / 20
```

Reset 时两者均为 0；当前 timestep routing 完成后更新，并只出现在返回的 next observation 中。因此当前 action 看到的是上一 timestep 已知的 execution feedback，没有 timing leakage。

其余配置保持不变：separate actor/critic、relative price、20 unit investors、Random competitor `U[-1,1]`、market sigma `0.2`、`2 x 256 tanh`、learning rate `5e-5`、PPO clip `0.3`、minibatch `256`、10 epochs、entropy `0.003`、相同 reward/PPO/tanh-Gaussian、相同 5 seeds 和 204,800 steps。Baseline 直接读取上一轮结果，没有重跑。

这里的两个 side-specific aggregates 是 paper previous-fill observation 的 compressed representation，不是逐 investor observation 的 exact replication。当前 investors homogeneous 且 order size 均为 1，因此没有增加 20 个高度重复的 identity inputs。

## 1. Mean-policy accuracy 是否改善？

**没有，final accuracy 明显变差。**

| Final metric | Relative-price baseline | + Fill feedback | Change |
|---|---:|---:|---:|
| Bid MAE | 0.1512 | 0.1690 | +11.7% |
| Ask MAE | 0.1690 | 0.2559 | +51.4% |
| Two-side MAE | 0.1601 | 0.2124 | +32.7% |

恶化主要来自 ask side。Fill feedback 并没有让最终 deterministic policy 更接近 Random competitor 的 analytical optimum `epsilon*=0`。

Final fill-feedback policies：

| Seed | det bid | det ask | Two-side MAE | Worst-side error |
|---:|---:|---:|---:|---:|
| 0 | -0.3990 | -0.3742 | 0.3866 | 0.3990 |
| 1 | -0.0203 | -0.1506 | 0.0855 | 0.1506 |
| 2 | 0.2640 | 0.4822 | 0.3731 | 0.4822 |
| 3 | 0.0838 | 0.0676 | 0.0757 | 0.0838 |
| 4 | -0.0778 | -0.2046 | 0.1412 | 0.2046 |

## 2. Seed stability 是否改善？

**没有实质性改善。** Per-seed two-side MAE dispersion 从 `0.1424` 降至 `0.1386`，只下降约 2.7%，不足以视为明显改善。同时 worst-side error 从 `0.4326` 增至 `0.4822`，说明 tail seed 没有改善。

Baseline 的主要 bad seed 是 seed 3；加入 fill feedback 后 seed 3 表现良好，但 seeds 0 和 2 分别形成负向和正向偏移。因此这更像 failure seed 的迁移，而不是 across-seed stability 提升。

## 3. 100k+ drift / oscillation 是否减轻？

**Early/mid-training 明显更稳定，但 long-run drift 没有消失。**

| Step | Baseline MAE | Fill-feedback MAE | Baseline dispersion | Fill-feedback dispersion |
|---:|---:|---:|---:|---:|
| 60,416 | 0.1930 | 0.1055 | 0.1780 | 0.0433 |
| 100,352 | 0.1807 | 0.1232 | 0.1979 | 0.0843 |
| 149,504 | 0.1561 | 0.1411 | 0.0722 | 0.0862 |
| 199,680 | 0.1828 | 0.1972 | 0.1330 | 0.1035 |
| 204,800 | 0.1692 | 0.1997 | 0.1491 | 0.1220 |

Fill-feedback group 在 60k-100k 确实有更低的 trajectory MAE 和 dispersion，但改善随后逆转：

- seed 0 从 149k 的 MAE `0.0930` drift 至 204.8k 的 `0.3915`，最终两侧约为 `-0.37/-0.41`；
- seed 2 从 100k 的 MAE `0.1960` 增至 149k 的 `0.3091`，ask side 在后期维持约 `0.39-0.44`；
- baseline seed 3 的 persistent drift 在新组中减轻，但并未形成 across-seed 的系统性改善。

所以 previous fills 可能改善短期 learning signal，却没有约束 long-run optimization drift。

## 4. Fill feedback 是否值得保留？

**以本轮证据，不值得作为当前默认 observation 保留。** 它没有改善 final MAE、worst-side error 或长期 seed stability。`include_fill_feedback=False` 继续保持默认值，因此历史实验和当前最佳 baseline 不受影响。

更准确的范围结论是：

> 在当前 homogeneous 20-unit-investor simulator 中，两个 compressed previous-fill fractions 不能解释剩余 replication gap。

这不等价于证明论文完整的逐 investor trade history 无效；本轮只否定了当前最小 compressed representation 在该 simulator/configuration 下的增量价值。

## Secondary metrics

Final latent sigma 仍接近 1：bid `0.9769 +/- 0.0066`，ask `0.9786 +/- 0.0093`。Market share 为 `0.5191 +/- 0.0808`，spread PnL / step 为 `0.1029 +/- 0.0178`。这些结果不改变 mean-policy 判断，也未展开额外 investigation。

## Validation

- Python compile：通过；
- Existing tests：8/8 通过；
- Existing diagnostics：通过；
- Short smoke run：通过；
- Timing sanity：reset fills 为 `[0, 0]`；当某一步 RL 赢得 11 个 bid fills 和 9 个 ask fills 时，next observation 为 `[0.55, 0.45]`。

本轮到此停止，不删除 PnL features、不增加其他 flow/market features，也不修改 PPO。
