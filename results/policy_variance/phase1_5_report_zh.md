# Policy Variance 与 Inventory-Skew 调查：Phase 1–5 阶段报告

## 结论摘要

1. `log_std` 并不缺少 gradient。其 pre-clip mean-absolute gradient 比 actor mean head 高约 `2.5×`，但 random 环境中的 signed gradient 在 minibatch 间大量抵消，导致 latent std 几乎不变。
2. Stochastic sampling 显著损害 quoting performance；deterministic mean policy 的 spread PnL 与 total PnL 明显更高。
3. Policy variance 可以解释 persistent competitor 原始 deterministic-best-response discrepancy 的大部分，因为有限 σ 下的公平 stochastic optimum 本来就远低于 `0.5⁻`。
4. 但 variance 不是唯一问题。`σ=0.1` 时 PPO 反而不能稳定学习对应 stochastic optimum，random competitor 的 mean policy 也仍有显著 seed instability。
5. Deterministic mean policy 含有弱且不稳定的正确 inventory-skew 信号：ask 侧较一致，bid 侧混合。跨 seed 直接 pooled 会产生 wrong-sign Simpson-type reversal。
6. 值得继续做小规模 entropy ablation 与 controlled state sweep/state-scaling audit；目前不值得进入 reward redesign。

## Phase 1：冻结 baseline

原 `results/replication/`、`results/risk_aversion/` 与 `docs/replication_findings.md` 未被覆盖。只读 baseline 摘要见 `baseline_snapshot.md`。Learned-variance baseline 的 bid/ask latent std 约 `0.97–1.00`，sampled epsilon std 约 `0.61–0.62`。

## Phase 2：Variance learning 与 gradient

所有 gradient 均在 global gradient clipping 前记录；actor mean gradient 指 policy mean head 参数的 mean absolute gradient。

| 环境 | final bid std | final ask std | final hedge std | mean `|grad(log_std)|` | mean `|grad(actor mean)|` | 比值 |
|---|---:|---:|---:|---:|---:|---:|
| Random | 0.9973 | 0.9973 | 0.9974 | 0.0674 | 0.0305 | 2.52 |
| Persistent spread-only | 0.9715 | 0.9721 | 0.9998 | 0.1077 | 0.0429 | 2.59 |

Random 的 bid/ask signed `log_std` gradient 均值约 `+0.007–0.008`，但标准差约 `0.04`，说明方向抵消严重。Persistent 的 bid/ask signed gradient 约 `+0.125`，方向较稳定，因此 std 有可见但缓慢的下降。Entropy-only gradient 是 `-0.003`：它阻碍 std 下降，但在 persistent 中远小于 policy signal；在 random 的小净梯度中相对更重要。

## Phase 3：Deterministic mean 与 stochastic policy

五 seeds 均使用同一 evaluation environment seeds。

| Random competitor | Deterministic | Stochastic |
|---|---:|---:|
| Mean bid epsilon | 0.0746 | 0.0478 |
| Mean ask epsilon | 0.2122 | 0.1416 |
| Bid epsilon std | 0.0075 | 0.6200 |
| Ask epsilon std | 0.0061 | 0.6075 |
| Market share | 0.4269 | 0.4530 |
| Spread PnL / step | 0.4218 | 0.2739 |
| Total PnL | 117.1 | 29.0 |
| Inventory std | 18.17 | 19.45 |
| Inventory-PnL std | 4.67 | 4.85 |

| Persistent spread-only | Deterministic | Stochastic |
|---|---:|---:|
| Mean bid epsilon | -0.0623 | -0.0368 |
| Mean ask epsilon | -0.1137 | -0.0683 |
| Market share | 1.000 | 0.741 |
| Spread PnL / step | 0.8513 | 0.4588 |
| Economic total PnL | 371.6 | 123.5 |

Random 环境中，per-seed deterministic slope 的平均值为：

- `d epsilon_bid/dz = +1.68e-4`；
- `d epsilon_ask/dz = -1.82e-4`；
- `d(epsilon_bid-epsilon_ask)/dz = +3.50e-4`。

方向正确，但幅度小且跨 seed 不稳。Ask slope 在 5/5 seeds 中方向正确；bid slope 只有 2/5 明显为正；综合 skew slope 4/5 为正。Pooled curve 的 wrong sign 来自 seed-level quote offsets 与 inventory distributions 的混合，不能代表单一 mean policy 的响应。

## Phase 4–5：Fixed-σ 与公平 stochastic optimum

### Random competitor

对 `epsilon_comp ~ U[-1,1]`，实际 stochastic objective 为 `E[(1-epsilon^2)/2]`，所有 σ 的 optimal mean epsilon 均为 `0`。

| σ | Learned deterministic bid | Learned deterministic ask | 两侧 MAE vs 0 | Sampled bid/ask std | Spread PnL / step | Market share |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.105 +/- 0.207 | 0.122 +/- 0.231 | 0.212 | 0.623 / 0.618 | 0.277 | 0.464 |
| 0.5 | 0.001 +/- 0.178 | 0.047 +/- 0.372 | 0.236 | 0.413 / 0.381 | 0.362 | 0.488 |
| 0.2 | 0.008 +/- 0.133 | 0.112 +/- 0.240 | 0.125 | 0.191 / 0.181 | 0.431 | 0.471 |
| 0.1 | 0.006 +/- 0.137 | 0.152 +/- 0.333 | 0.161 | 0.098 / 0.086 | 0.426 | 0.460 |

降低 σ 会缩窄 sampled distribution，但 learned mean 的 seed instability 不单调消失；`σ=0.2` 最好，`σ=0.1` 反而受个别失败 seed 影响。

### Persistent competitor，spread-only

| σ | Stochastic optimal epsilon | Learned deterministic bid | Learned deterministic ask | 两侧 MAE | Stochastic market share | Spread PnL / step |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | -0.056 | -0.062 +/- 0.073 | -0.116 +/- 0.085 | 0.082 | 0.735 | 0.448 |
| 0.5 | 0.056 | 0.058 +/- 0.037 | 0.018 +/- 0.035 | 0.036 | 0.844 | 0.723 |
| 0.2 | 0.232 | 0.134 +/- 0.119 | 0.188 +/- 0.053 | 0.082 | 0.959 | 1.019 |
| 0.1 | 0.338 | 0.050 +/- 0.259 | 0.137 +/- 0.143 | 0.254 | 0.989 | 1.005 |

`σ=1.0/0.5` 支持“PPO 正在学习 stochastic-policy optimum”；`σ=0.2` 只部分支持；`σ=0.1` 明确不支持。低 σ 虽提高 market share，但使 threshold objective 的有效 learning region 更窄，mean optimization 出现明显 seed failures。

## 对当前假设的判断

### Policy variance 能否解释 best-response discrepancy？

**能解释大部分原始 persistent discrepancy，但不能解释全部。** 有限 σ 下，公平 benchmark 本来就接近 0 而不是 `0.5⁻`。但在 σ 很低时，PPO 仍可能偏离对应 stochastic optimum，因此还存在 mean optimization/stability 问题。

### Deterministic mean 是否已经包含正确 inventory skew？

**包含弱信号，但尚不稳健。** Ask side 较一致，bid side 混合；跨 seed pooled wrong sign 是 confounding，但 per-seed mean policy 也没有形成强而一致的 skew。

### 是否值得继续调查？

- **Entropy：值得。** 做 Phase 6 的小型 ablation 可以量化 random 环境中 `-0.003` entropy gradient 的实际贡献；预计它不是 persistent variance 的主因。
- **State scaling：值得，但先做 controlled state sweep。** Mean policy 的弱 slope 与 seed-level quote offsets 使 state sensitivity/saturation 成为合理怀疑；应先执行 Phase 9，再决定是否进入 Phase 10 normalization。
- **Reward redesign：暂不值得。** 当前证据尚未排除 policy optimization 与 state representation；直接改 reward 会混淆诊断。

## 本阶段停止点

按任务要求，本轮在 Phase 5 后停止。未执行 entropy ablation、freeze-mean/variance-only、reduced-variance full-environment skew、controlled state sweep、state scaling 或 reward diagnosis。
