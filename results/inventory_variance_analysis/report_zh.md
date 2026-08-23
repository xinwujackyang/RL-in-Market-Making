# Inventory vs State-Dependent Variance

## 结论

本实验属于 **Case A：存在明显且跨 seed 的 inventory pattern**。

在 Persistent state-dependent candidate baseline 的 final stochastic evaluation states 上，`σ_bid(s)-σ_ask(s)` 随 action 前 inventory 从低到高呈现清晰变化：低 inventory bins 通常为负，高 inventory bins 通常为正。Seeds 0、1、2、3 的五-bin delta 均严格单调上升；seed 4 基本平坦且略向下，是唯一不支持该 pattern 的 seed。

因此：

> **Inventory is an important state variable through which the policy reallocates exploration across bid and ask.**

这是一项 conditional association diagnostic，不证明因果，也不解释网络是否同时使用 relative price 或 PnL。本轮按计划到此停止，不扩展其他 state features。

## 方法

- 重新训练 Persistent competitor `(0.5, 0.5, 0)` 下的 5 个 `state_dependent_std=True` seeds，每个 204,800 steps。
- 每个 seed 使用 final stochastic evaluation 的 5,200 个实际访问 states。
- 只保留 `inventory_before_action`、`latent_std_bid`、`latent_std_ask` 三个内存 arrays。
- 将 5 seeds 共 26,000 个 inventory observations pooled 后定义共同的 20/40/60/80% quantile edges。
- 在每个 seed 内分别计算五个 common bins 的 inventory mean、bid/ask sigma mean、`σ_bid-σ_ask` 与 count。
- 不保存逐步 raw trajectories，不运行 global-std 组。

## Cross-seed conditional means

下表先在每个 seed/bin 内求 mean，再对 5 seeds 等权平均：

| Inventory bin | Inventory mean | Sigma bid | Sigma ask | Bid minus ask |
|---|---:|---:|---:|---:|
| Q1（lowest 20%） | -11.269 | 0.632 | 1.381 | -0.749 |
| Q2 | -3.964 | 0.622 | 1.085 | -0.463 |
| Q3 | 0.483 | 0.777 | 0.648 | 0.129 |
| Q4 | 5.116 | 1.156 | 0.572 | 0.584 |
| Q5（highest 20%） | 12.671 | 1.362 | 0.622 | 0.740 |

Pattern 不只是 total variance 随 inventory 改变，而是 bid/ask exploration allocation 发生反转：

- Low-inventory states：ask sigma 明显高于 bid sigma。
- Near-zero inventory：两侧 sigma 接近，average delta 略为正。
- High-inventory states：bid sigma 明显高于 ask sigma。

## Per-seed robustness

| Seed | Q1 delta | Q2 delta | Q3 delta | Q4 delta | Q5 delta | Q1→Q5 pattern |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -1.127 | -0.734 | 0.051 | 1.026 | 1.545 | 严格上升 |
| 1 | -0.377 | -0.262 | 0.260 | 0.840 | 0.993 | 严格上升 |
| 2 | -1.448 | -0.960 | 0.095 | 0.596 | 0.673 | 严格上升 |
| 3 | -0.878 | -0.428 | 0.157 | 0.391 | 0.464 | 严格上升 |
| 4 | 0.084 | 0.070 | 0.083 | 0.065 | 0.022 | 近似平坦、略下降 |

4/5 seeds 独立表现出相同方向和严格 monotonicity，因此 pooled pattern 不是由单一 seed 驱动。Seed 4 的 variance 仍随 state 变化，但 inventory bins 不能解释其 bid/ask allocation。

## Decision

Inventory dependence 足够 robust，值得作为 state-dependent variance mechanism 的已确认特征记录下来。但本轮不继续 relative-price、PnL、interaction、regression 或 feature attribution 分析。

## Validation

- 5/5 seeds 完成，每 seed 5,200 个 stochastic-evaluation observations。
- 每个 seed 都包含 inventory-before-action 与 bid/ask latent std arrays，三者长度一致。
- 26,000 observations 全部且仅分入一个 common quantile bin；每个 pooled bin 恰有 5,200 observations。
- 输出为 5 seeds × 5 bins，共 25 rows，无 NaN/Inf。
- Python compile 通过；`src/ppo.py`、`src/networks.py`、`src/evaluation.py` 均未修改。
