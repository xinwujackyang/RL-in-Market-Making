# Remove Non-Paper PnL Features Experiment

## 结论

从 relative-price observation 中同时删除 `TotalPnL` 和 `HedgeCost`，只保留 `[inventory, relative price, inventory PnL]`，**没有改善 long-run mean-policy stability，反而造成更严重的 bad-seed failure**：

- Final deterministic two-side MAE：`0.1601 -> 0.1888`，恶化 18.0%；
- Per-seed MAE dispersion：`0.1424 -> 0.2244`，恶化 57.5%；
- Worst-side error：`0.4326 -> 0.6768`，恶化 56.5%；
- seed 2 在 100k-150k 之间突然 drift 到较大的正 epsilon，并持续到训练结束。

因此本轮支持：

> **不能把 `TotalPnL` / `HedgeCost` 视为已证实的 nuisance features。三维 state 虽然在 early training 更接近 analytical optimum，但 long-run tail stability 明显更差，应保留当前五维 relative-price baseline observation。**

按照停止条件，本轮不继续单独 ablate 两个 features，也不猜测其他 feature removal。

## 实验控制

唯一主动变化：

```text
Baseline: [inventory, relative price, TotalPnL, inventory PnL, HedgeCost]
New:      [inventory, relative price, inventory PnL]
```

新组明确关闭 fill feedback。其余配置保持不变：separate actor/critic、20 unit investors、Random competitor `U[-1,1]`、market sigma `0.2`、`2 x 256 tanh`、learning rate `5e-5`、PPO clip `0.3`、minibatch `256`、10 epochs、entropy `0.003`、相同 reward/PPO/bounded tanh-Gaussian、相同 5 seeds 和 204,800 steps。Baseline 直接读取已有 relative-price 结果，没有重跑。

## 1. Two-side MAE：0.1601 -> 0.1888

**没有改善，平均误差增加约 18.0%。**

| Final metric | Five-feature baseline | Three-feature state | Change |
|---|---:|---:|---:|
| Bid MAE | 0.1512 | 0.1888 | +24.9% |
| Ask MAE | 0.1690 | 0.1888 | +11.7% |
| Two-side MAE | 0.1601 | 0.1888 | +18.0% |

Final three-feature policies：

| Seed | det bid | det ask | Two-side MAE | Worst-side error |
|---:|---:|---:|---:|---:|
| 0 | 0.0796 | -0.1541 | 0.1168 | 0.1541 |
| 1 | -0.1353 | 0.0199 | 0.0776 | 0.1353 |
| 2 | 0.5909 | 0.6768 | 0.6339 | 0.6768 |
| 3 | 0.0011 | 0.0542 | 0.0276 | 0.0542 |
| 4 | -0.1373 | -0.0393 | 0.0883 | 0.1373 |

只有 baseline 的 bad seed 3 明显改善；seeds 0、1、2、4 的 final MAE 均变差，其中 seed 2 的恶化远大于 seed 3 的改善对 stability 的帮助。

## 2. Seed dispersion：0.1424 -> 0.2244

**明显恶化。** Per-seed two-side MAE dispersion 增加约 57.5%。四个 seeds 的 final MAE 在约 `0.028-0.117`，但 seed 2 达到 `0.6339`，形成更重的 tail。

这说明三维 state 可能让多数 seed 更容易形成接近 0 的 policy，但不能稳定避免 catastrophic optimization path。以本轮五个 paired seeds，不能将其描述为更稳定的 state representation。

## 3. Worst-side error：0.4326 -> 0.6768

**明显恶化。** Worst-side error 增加约 56.5%，来自 seed 2 的 ask `0.6768`；其 bid 也达到 `0.5909`。Baseline 的 extreme seed 并未真正消失，而是从 seed 3 迁移为更严重的 seed 2 failure。

## 4. 100k+ drift 是否减轻？

**没有。Early trajectory 改善，但 100k 后出现更严重的 persistent drift。**

| Step | Baseline MAE | Three-feature MAE | Baseline dispersion | Three-feature dispersion |
|---:|---:|---:|---:|---:|
| 60,416 | 0.1930 | 0.1571 | 0.1780 | 0.0472 |
| 100,352 | 0.1807 | 0.1348 | 0.1979 | 0.0720 |
| 149,504 | 0.1561 | 0.2517 | 0.0722 | 0.2477 |
| 199,680 | 0.1828 | 0.2021 | 0.1330 | 0.2537 |
| 204,800 | 0.1692 | 0.1945 | 0.1491 | 0.2458 |

三维组在 100k 前的 aggregate MAE 和 dispersion 都更低。但 seed 2 的 deterministic policy 从 100k 的约 `(0.072, 0.341)` 快速 drift 到 149k 的 `(0.704, 0.787)`，之后一直维持大幅正偏移，204.8k 仍约为 `(0.643, 0.722)`。

其余 seeds 没有同类 runaway，说明问题仍然是 seed-dependent long-run instability，而不是所有 policy 的统一 bias。

## 5. 删除 TotalPnL / HedgeCost 是否值得保留？

**不值得。** Final MAE、seed dispersion 和 worst-side error 三项主要指标全部恶化，因此 `paper_pnl_observation=False` 应继续作为默认设置，并保留当前五维 relative-price baseline。

本轮只能判断两个 features 作为 conceptual group 的联合删除无益；按照实验边界，不进一步拆分为单-feature ablation。

## Secondary metrics

Final latent sigma 仍接近 1：bid `0.9831 +/- 0.0081`，ask `0.9810 +/- 0.0075`。Market share 为 `0.4767 +/- 0.0884`，spread PnL / step 为 `0.1041 +/- 0.0233`。这些 secondary metrics 不改变 mean-policy stability 结论。

## Validation

- Observation dimension：`5 -> 3`；
- Feature order：`inventory, relative price, inventory PnL`；
- Python compile：通过；
- Existing tests：9/9 通过；
- Existing diagnostics：通过；
- Short smoke run：通过。

本轮到此停止，不继续其他 observation feature 修改。
