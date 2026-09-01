# RL Market Making：最终研究总结

## 1. Project question

本项目研究的问题是：

> How does a PPO market maker learn quoting, inventory control, and hedging under competitive dealer markets, and how does it compare with heuristic, adaptive, and classical stochastic-control benchmarks?

项目从 Ganesh et al. (2019) 启发的 PPO 复现出发，逐步转向一个更可检验的研究程序：先审计 simulator 和 PPO correctness，再分析 learned policy，最后用固定 heuristic、online Adaptive MM 和 classical Avellaneda–Stoikov controller 回答不同层次的问题。本项目是 controlled behavioral investigation，不是原论文的数值复现。

| Method | 类型 | Learning mechanism | Inventory control | External hedge |
|---|---|---|---|---|
| Persistent | heuristic | No | none | No |
| Adaptive | empirical online learning | response table | quote skew | Yes |
| A-S | stochastic control | No | analytical skew | No |
| PPO | model-free RL | neural policy | learned skew | Yes |

这四类方法没有被拼成一个误导性的全模型 PnL leaderboard；每项 comparison 对应独立的 research question 和匹配的 protocol。

## 2. Simulator

核心环境是双 dealer 市场。每一步中，dealer 先决定

```text
(epsilon_bid, epsilon_ask, hedge_fraction)
```

然后依次发生：已有库存 hedge、investor orders routing、库存更新、GBM mid-price evolution，以及 post-flow inventory PnL realization。报价相对 simulator reference spread 参数化；较小 epsilon 表示更 aggressive 的价格。PnL 恒等式为：

```text
TotalPnL = SpreadPnL + InventoryPnL - HedgeCost
```

环境使用 dealer-perspective inventory sign，并为 order size、direction、price 和 routing 保持独立 RNG streams。正式 Adaptive/A-S comparisons 使用 20 个 unit orders/step、buy probability 0.5、sigma=0.2。

最重要的 execution 限制是 winner-take-all：每笔 order 全部路由给更优 quote，完全 tie 才随机平分。它适合做 controlled competition，但不是 continuous LOB fill model，也不能识别 textbook A-S 的 smooth exponential arrival elasticity。

## 3. PPO reference implementation

在解释策略之前，项目系统修正和验证了：event timing、inventory signs、PnL accounting、GAE rollout-boundary bootstrap、PPO clipped ratio，以及 bounded action likelihood。最终 reference PPO 使用：

- 5D observation：inventory、relative price、total PnL、inventory PnL、hedge cost；
- separate `2 x 256` tanh actor/critic；
- state-dependent tanh-squashed Gaussian policy；
- Jacobian-corrected transformed log probability；
- action support：quotes `[-1, 1]`，hedge `[0, 1]`；
- 204,800-step long-run budget。

两个关键工程结论是 observation scale 和 exploration parameterization 很重要。Relative-price feature 显著缓解 actor saturation；state-dependent policy variance 相比 global variance 将 Random benchmark 的 final deterministic two-side MAE 从 0.1601 降到 0.0808，并将 seed dispersion 从 0.1424 降到 0.0467。这些结果构成 reference policy 的选择依据，但不是 hyperparameter optimality claim。

详细 correctness audit：[`docs/correctness_audit.md`](../../docs/correctness_audit.md)。

## 4. Adaptive benchmark

Adaptive MM 使用离散 joint quote response table，在线更新 gross volume、net flow 和 normalized spread PnL moments；它用 symmetric quote targeting market share，再用 correcting-side skew 和 hedge grid 管理库存。

为了避免正式 competition 被 121-cell cold start 主导，每个 seed 先进行 10 次完整 grid calibration，共 1,210 steps，再在 fresh formal environment 中从同一 PPO initialization 开始。正式 long run 使用 probe interval=20，即 5% diagonal probes。Persistent diagonal probing 是本项目为了在 learning opponent 下维持 response-map tracking 引入的机制，不是原论文明确定义的 canonical algorithm。

在 204,800-step PPO-vs-Adaptive run 中：

- PPO market share 从 100,352 的 45.60% 上升到 204,800 的 54.16%；
- PPO normalized spread monetization 相对 Adaptive 的 ratio 为 `1.1218 -> 1.2001 -> 1.3447`；
- final PPO total PnL/step 为 0.4784，Adaptive 为 0.2985；PPO 在 3/3 seed paths 胜出；
- final mean PnL advantage 0.1799 分解为 spread `+0.1970`、inventory `-0.0170`、hedge-cost difference `-0.0001`。

因此 PPO 后期同时改善了 margin 和 volume operating point，优势来自 spread capture，而不是 inventory speculation 或更低 hedge cost。但这不是 convergence result：100k 到 204.8k 的 policy/economics 仍明显变化，PPO total-PnL cross-seed std 从 0.0332 扩大到 0.3906。

这里的 normalized monetization 受已保存 artifact 边界约束，定义为 `(window aggregate spread PnL / captured volume) / window mean S_ref(1)`，不是无法从现有 aggregate 恢复的 dealer-volume-weighted estimator。它足以排除 nominal price scale 对 raw spread/unit 上升的简单解释，但不应被描述成更精细的 fill-level normalization。

详细结果：[`results/ppo_vs_adaptive_probe20_longrun/report_zh.md`](../ppo_vs_adaptive_probe20_longrun/report_zh.md) 与 [`results/ppo_vs_adaptive_normalized_spread/report_zh.md`](../ppo_vs_adaptive_normalized_spread/report_zh.md)。

## 5. A-S benchmark

A-S controller 保留 canonical structure：

```text
r = S - q gamma V
Delta = gamma V + 2/gamma log(1 + gamma/k)
```

但正式 benchmark 是 normalized stationary compatibility implementation，而不是原始 exponential Poisson-arrival model 的直接 calibration。Compatibility choices 为：

- 26-step、one-trading-day receding risk horizon；
- local GBM/Brownian variance approximation；
- fixed normalized `rho`、`kappa`，dollar equivalents 随 reference spread 缩放；
- neutral quote 对齐 unit reference quote；
- `inventory_anchor=20`，即 20 units 对应一个 full reference-distance shift；
- native winner-take-all execution；
- zero external hedge；
- executable quote clipping 及显式 clip statistics。

Standalone A-S vs Persistent 在 3 seeds x 100,352 steps 中得到 `E|q|=4.998`，Persistent 为 1131.933；正库存 conditional flow 为负、负库存 conditional flow 为正，side clipping 为 0%。这些结果验证 analytical skew 在当前 native routing 中确实产生库存 mean reversion，并冻结了 benchmark 参数，没有进行 PnL-based tuning。

详细结果：[`results/as_benchmark/report_zh.md`](../as_benchmark/report_zh.md)。

## 6. Learned PPO behavior

将 quote action 分解为：

```text
m = (epsilon_bid + epsilon_ask) / 2
k = epsilon_bid - epsilon_ask
```

其中 `m` 是 symmetric quote level，控制整体 margin/volume；`k` 是 relative skew，控制 inventory liquidation direction。

在 Adaptive experiment 的 100,352-step policy analysis 中，final `beta_k > 0` 出现在 3/3 seeds，说明 PPO 一致学到 economically signed inventory-dependent skew。PPO 的 hedge fraction 本身不随 `|q|` 单调增加，但 absolute hedge quantity `E[h|q|]` 在库存 bins 中为 `0.2816 -> 1.5599 -> 3.1131 -> 4.6862`，因此策略确实包含 absolute balance-sheet reduction；同时 residual exposure 仍增加，不能表述为比例上完全 neutralize inventory。

对 frozen A-S 的 final deterministic analysis 给出：

```text
A-S: k_AS(q) = 0.1 q                         before clipping
PPO: k_PPO(q) ≈ 0.1255 + 0.0476 q           pooled R² = 0.578
```

在 pooled `|q|>=1` states 中，96.0% 的 PPO skew 具有与 A-S 相同的 corrective direction；三个 seed 的 slope 都为正。不过 PPO sensitivity 更弱、明显 nonlinear/state-dependent，且单 seed corrective-sign frequency 为 77.6%–100%。准确结论是 PPO independently rediscovers the direction of A-S-like inventory control，而不是“PPO learned A-S”。

详细结果：[`results/ppo_behavior_probe20/report_zh.md`](../ppo_behavior_probe20/report_zh.md) 与 [`results/ppo_vs_as/report_zh.md`](../ppo_vs_as/report_zh.md)。

## 7. Economic comparisons

不同 benchmark 不共享全部 opponent、training 和 evaluation protocols，因此应按问题阅读：

| Research question | Comparison | Main result |
|---|---|---|
| Does PPO learn against a fixed dealer? | PPO vs Persistent | learns nontrivial quoting and inventory-conditioned exploration |
| Can PPO compete with an adaptive dealer? | PPO vs Adaptive | eventually improves both normalized margin and volume |
| Does PPO rediscover classical inventory control? | PPO vs A-S | yes in direction, not in exact functional form |
| Does richer RL beat A-S economically? | PPO vs A-S | no; A-S wins 2/3 seed paths |

PPO-vs-A-S 的 final deterministic evaluation 为：

| Metric | PPO | A-S | PPO - A-S |
|---|---:|---:|---:|
| Total PnL/step | 0.1340 ± 0.0796 | 0.2388 ± 0.1252 | -0.1048 |
| Spread PnL/step | 0.1367 ± 0.0716 | 0.2322 ± 0.1289 | -0.0955 |
| Inventory PnL/step | 0.0131 ± 0.0058 | 0.0066 ± 0.0271 | +0.0065 |
| Hedge cost/step | 0.0158 ± 0.0040 | 0 | +0.0158 |

因此 gap 精确分解为：

```text
-0.1048 = -0.0955 + 0.0065 - 0.0158
```

PPO 只在 1/3 seed paths 胜出。负结果的主要原因是 lower spread capture 加上 nonzero hedge cost，而不是 inventory loss；更丰富的 neural control space 并未在这个 protocol 下击败 classical A-S compatibility controller。

## 8. Main conclusions

1. **PPO 学到了 economically meaningful MM behavior。** Learned policy 同时控制 symmetric quote level、inventory-dependent relative skew 和 absolute hedge quantity。
2. **Against Adaptive，PPO 后期改善了 margin-volume trade-off。** 204.8k 时 share 超过 50%，normalized monetization advantage 扩大，并在 3/3 seeds 获得更高 PnL；但尚无 plateau，不能声称 convergence。
3. **PPO independently rediscovers A-S-like inventory control。** Direction 高度一致，但 pooled slope 更弱，且有明显 nonlinear/state-dependent departures。
4. **Richer RL 没有 beat classical A-S。** A-S 在 2/3 seeds 胜出，平均优势主要来自 spread capture 和 zero hedge cost。保留这一 negative result 比继续 tuning competitor 更有研究价值。

更广泛的工程结论是：correct semantics、observation representation 和 exploration parameterization 的影响，往往比局部 KL、critic loss 或短期 training curve 更可靠；multi-seed long-horizon evidence 仍然是解释 learned market-making policy 的必要条件。

## 9. Limitations

- **Winner-take-all execution。** Best quote wins the entire investor order；没有 probabilistic partial fills、queue position 或 continuous LOB dynamics。
- **A-S compatibility model。** 使用 one-day receding horizon、normalized `rho/kappa`、`inventory_anchor=20`、native routing 和 zero hedge；不是原始 Poisson-arrival A-S 的 empirical calibration。
- **Adaptive probing。** Persistent diagonal probing 是 project-specific tracking mechanism，不应描述成 paper canonical algorithm。
- **No convergence claim。** Adaptive long-run 到 204.8k 仍在变化且 dispersion 扩大；这里只能称为 observed long-run regime。
- **Simulation scope。** 单一 volatility、unit flow、单 competitor、有限 observation 和 synthetic GBM price 限制了 external validity。
- **Cross-benchmark comparability。** Persistent、Adaptive、A-S 数字来自不同 research protocols，不能组成统一 PnL leaderboard。

## 10. What I would do next

项目在当前 benchmark scope 下冻结。若未来有明确的新 research question，优先方向是：

1. smooth/probabilistic execution 与 partial-fill model；
2. 用真实 market microstructure 校准 spread、flow 和 volatility；
3. opponent cross-play 与 out-of-distribution generalization；
4. 带 external hedge 的 richer A-S extension；
5. real-data 或 replay-based evaluation。

这些是 future research directions，不是当前 repository 的待办清单。当前项目的结论应以已提交 artifacts 和固定 protocols 为边界。
