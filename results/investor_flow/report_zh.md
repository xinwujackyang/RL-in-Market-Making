# Investor Flow A/B：Random Competitor 五种子结果

## 结论

在当前训练预算下，把默认的 20 个 Gamma-size investors 改为论文式 20 个 unit-size investors，**没有改善 PPO 的 competitor-pricing learning，也没有降低真正的 policy variance 或改善 seed stability**。因此本轮没有触发完整 paper-PPO configuration；这是预先规定的停止条件，而不是遗漏实验。

四个问题的直接答案：

1. **20-investor unit-flow 是否显著降低 policy variance？否。** Bid latent std 从 `0.9979` 变为 `0.9968`，ask 从 `0.9978` 变为 `1.0007`，实质不变。Sampled bid std 略升；sampled ask std 的表观下降由一个 ask mean 饱和的失败 seed 驱动，并非 latent variance 学习。
2. **是否显著改善 Random competitor 下的 seed stability？否。** Bid pricing mean 的 across-seed std 几乎不变（`0.1624 -> 0.1623`），ask 反而从 `0.1550` 增至 `0.2394`。
3. **变化来自 mean-policy learning 还是 action variance reduction？主要是 mean-policy 的 seed-dependent 位移，而且净效果更差；没有 action-variance reduction。** Unit flow 的 bid error 略降，但 ask error 明显上升；latent std 全程停留在约 `1.0`。
4. **完整 paper PPO 是否进一步接近论文？本轮未测试，因触发条件不成立，不能作肯定或否定判断。** 按任务停止规则，只有 Flow-Only 明显改善后才运行 paper PPO；本结果明确没有改善。

## 实验设计

Flow-Only A/B 使用完全相同的 5 个 seeds、Random competitor `U[-1,1]`、PPO、network、learning rate、clip、rollout budget、entropy coefficient 和 mid-price 参数。每个 seed 训练 `20 x 1024 = 20,480` steps，最终用 10 个 20-day episodes（每 episode 520 steps）评估 stochastic policy。

当前 PPO 对齐此前 Random replication：3 hidden layers x 256、tanh、learning rate `5e-5`、clip `0.2`、minibatch `256`、entropy coefficient `0.003`。两组都保留 corrected sign/timing/accounting 和 bounded tanh-Gaussian PPO。

为了使 paired A/B 真正共享外生路径，size、buy/sell direction、mid-price 和 tie routing 使用由同一个 seed 派生的独立 RNG streams。因此两种 flow 在相同 seed 下拥有完全一致的方向序列和 mid-price path；唯一的 flow 差异是 order size。

一个重要限定是：当前 Gamma baseline **原本已经是每步 20 个独立订单，并逐单 routing**。因此本 A/B 实际隔离的是：

```text
20 x Gamma(shape=2, scale=1) size
vs
20 x unit size
```

它不是“单个 aggregate investor vs 20 investors”的比较。

## Policy learning

以下 `+/-` 均为 5 seeds 的 mean 和 population std。Pricing mean 和 action std 来自 stochastic policy evaluation。

| Flow | E[epsilon_bid] | E[epsilon_ask] | Std(epsilon_bid) | Std(epsilon_ask) | latent sigma_bid | latent sigma_ask |
|---|---:|---:|---:|---:|---:|---:|
| Gamma baseline | 0.1465 +/- 0.1624 | 0.1088 +/- 0.1550 | 0.6080 +/- 0.0252 | 0.6105 +/- 0.0187 | 0.9979 +/- 0.0044 | 0.9978 +/- 0.0025 |
| Unit paper flow | 0.1100 +/- 0.1623 | 0.2543 +/- 0.2394 | 0.6114 +/- 0.0288 | 0.5717 +/- 0.0933 | 0.9968 +/- 0.0034 | 1.0007 +/- 0.0015 |

Ask sampled std 的平均值下降 `0.0388`，但这不是稳定的 variance 学习。前四个 seeds 的 ask std 均值约为 `0.6195`（Gamma）和 `0.6181`（unit），几乎相同；差异主要来自 unit seed 4，其 sampled ask mean 为 `0.7247`、std 为 `0.3858`。这是 tanh action 接近上界后压缩 observed std 的结果，而该 seed 的 latent ask std 仍为 `0.9997`。

训练轨迹也给出同一结论：20,480 steps 后两组 latent std 仍约为初始值 `1.0`，没有形成持续下降。

![Latent std training trajectory](/Users/jackyang/Desktop/Courses_IUB/Research/RL/RLMM/results/investor_flow/latent_std_trajectory_flow_only.png)

## Analytical error 与 seed stability

Random competitor 的 analytical benchmark 是 `epsilon*=0`。

| Seed | Gamma bid error | Gamma ask error | Unit bid error | Unit ask error |
|---:|---:|---:|---:|---:|
| 0 | 0.0703 | 0.1241 | 0.0813 | 0.0831 |
| 1 | 0.3823 | 0.0218 | 0.3967 | 0.1265 |
| 2 | 0.1076 | 0.1463 | 0.0366 | 0.2182 |
| 3 | 0.1662 | 0.1572 | 0.1320 | 0.1190 |
| 4 | 0.2213 | 0.3425 | 0.0967 | 0.7247 |
| **Mean +/- seed std** | **0.1895 +/- 0.1092** | **0.1584 +/- 0.1038** | **0.1487 +/- 0.1277** | **0.2543 +/- 0.2394** |

Bid mean error 改善 `0.0409`，但 error dispersion 增加；ask mean error 恶化 `0.0959`，dispersion 增加超过两倍。两侧合并 MAE 从约 `0.1740` 增至 `0.2015`。因此不能把 unit flow 描述为更稳定或更接近 analytical optimum。

Deterministic mean policy 也显示同样模式：Gamma 的 bid/ask mean 为 `0.2322/0.1656`，unit 为 `0.1737/0.3530`。主要变化是 mean-policy 位移，不是 learned variance 收缩。

## Environment diagnostics

每个单元格仍是 across-seed mean +/- seed std；其中 timestep std 是先在每个 seed 的 evaluation timesteps 内计算 std，再跨 seed 汇总。

| Diagnostic | Gamma baseline | Unit paper flow |
|---|---:|---:|
| N_buy mean / timestep std | 10.0115 +/- 0.0350 / 2.2245 +/- 0.0245 | 同左 |
| N_sell mean / timestep std | 9.9885 +/- 0.0350 / 2.2245 +/- 0.0245 | 同左 |
| N_RL won mean / timestep std | 8.7518 +/- 1.1280 / 7.1041 +/- 0.1253 | 8.2150 +/- 0.9203 / 6.9709 +/- 0.3583 |
| N_competitor won mean / timestep std | 11.2482 +/- 1.1280 / 7.1041 +/- 0.1253 | 11.7850 +/- 0.9203 / 6.9709 +/- 0.3583 |
| Gross investor volume mean / timestep std | 39.9702 +/- 0.1089 / 6.3819 +/- 0.0532 | 20.0000 / 0.0000 |
| Net investor flow mean / timestep std | 0.1263 +/- 0.2065 / 10.9076 +/- 0.1367 | 0.0230 +/- 0.0701 / 4.4490 +/- 0.0489 |
| RL market share mean / timestep std | 0.4378 +/- 0.0565 / 0.3592 +/- 0.0065 | 0.4108 +/- 0.0460 / 0.3485 +/- 0.0179 |
| Spread PnL/step mean / timestep std | 0.2732 +/- 0.0081 / 0.3316 +/- 0.0110 | 0.1243 +/- 0.0121 / 0.1508 +/- 0.0039 |

Unit flow 完全符合 `N_buy ~ Binomial(20, 0.5)`：observed timestep std `2.2245` 接近理论值 `sqrt(5)=2.2361`。其 gross volume 恒为 20，net-flow std `4.4490` 接近理论值 `sqrt(20)=4.4721`。Gamma baseline 的 gross volume 约 40，net-flow std 约 10.91；unit mode 的确大幅降低环境流量尺度与噪声。

但这没有提供新的 routing resolution。每个 timestep 内 dealer quotes 固定，因此所有 buys 使用同一 ask comparison，所有 sells 使用同一 bid comparison；除 ties 外，同方向订单整体去同一个 dealer。Gamma baseline 已经有同样的 20-order、two-side routing 结构。Unit mode 只是让 order-count market share 落在 `1/20` 网格，并把 volume weighting 改为 equal weighting，并未增加独立 quote comparisons 的数量。这解释了为何 flow noise 下降，却没有改善 policy-gradient variance learning。

## Secondary market metrics

| Flow | Market share | Spread PnL/step | Inventory std | Total PnL / episode | Total PnL/step |
|---|---:|---:|---:|---:|---:|
| Gamma baseline | 0.4378 +/- 0.0565 | 0.2732 +/- 0.0081 | 18.3248 +/- 0.7331 | 39.4701 +/- 13.8600 | 0.0759 +/- 0.0267 |
| Unit paper flow | 0.4108 +/- 0.0460 | 0.1243 +/- 0.0121 | 9.0640 +/- 0.5717 | 15.6940 +/- 15.1650 | 0.0302 +/- 0.0292 |

PnL 和 inventory 的下降主要反映 gross volume 约减半，不应被解释为 policy learning 的改善或恶化。核心结论仍应来自 pricing mean、sampled action std、latent std 和 across-seed analytical error。

![Flow-only comparison](/Users/jackyang/Desktop/Courses_IUB/Research/RL/RLMM/results/investor_flow/comparison_flow_only.png)

## 停止决定

Flow-Only A/B 未满足“Paper Flow 明显改善”的触发条件，因此没有运行 `unit_current_ppo vs unit_paper_ppo`。本轮也没有进入 Persistent、entropy ablation、inventory sweep、Adaptive MM 或 reward redesign。

