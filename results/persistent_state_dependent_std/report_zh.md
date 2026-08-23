# Persistent Competitor — State-Dependent Std Validation

## 结论

State-dependent std 的稳定性改善 **总体上可以跨 competitor generalize，但改善具有明显 side asymmetry**。

Persistent `ε_bid=ε_ask=0.5` 下，final deterministic bid 的 cross-seed dispersion 下降 `54.8%`，ask dispersion 下降 `5.6%`；两侧 dispersion 的简单平均下降 `35.1%`。Global-std seed 3 在 60k 后形成的双侧正向 runaway 被消除，但 state-dependent seeds 0/2 仍存在较明显的负向 ask drift。因此 bad-seed risk 明显减少但未完全消失，最强证据来自 bid side。

经济结果没有恶化：market share、spread PnL / step、total PnL / step 和 inventory std 四项 aggregate metrics 全部改善。Persistent 环境中 `σ(s)` 也保持明显的 state dependence。综合 policy stability 与 economic sanity checks，支持：

> **State-dependent exploration improvement generalizes across competitors.**

因此将 `state_dependent_std=True` 正式作为后续实验 baseline；不继续调整 std。

## Matched comparison

本实验重新运行两组完全 matched 的 Persistent experiments：

- Global std：`state_dependent_std=False`
- State-dependent std：`state_dependent_std=True`

两组均为相同 5 seeds、每 seed 204,800 training steps，并共享相同 simulator/evaluation seed streams。其余配置固定为 5D relative-price observation、separate actor/critic、2×256 tanh、20 unit investors、market sigma `0.2`、learning rate `5e-5`、PPO clip `0.3`、gamma `0.999`、GAE λ `0.95`、minibatch 256、10 epochs、entropy `0.003`、value coefficient `0.5`、global grad clip `0.5`、MC critic target、无 value clipping、无 KL。

本报告不计算到 `0.5` 的 MAE；deterministic bid/ask means 仅作为 learned-policy location 与 cross-seed stability 的描述。

## 1. Cross-seed stability

| Final deterministic policy | Global std | State-dependent std | 变化 |
|---|---:|---:|---:|
| Bid mean across seeds | -0.0257 | -0.0776 | — |
| Bid seed dispersion | 0.3881 | 0.1753 | -54.8% |
| Ask mean across seeds | -0.1455 | -0.2585 | — |
| Ask seed dispersion | 0.2603 | 0.2456 | -5.6% |
| Mean of bid/ask dispersion | 0.3242 | 0.2105 | -35.1% |

Per-seed final deterministic policies：

| Seed | Global bid | Global ask | State-dependent bid | State-dependent ask |
|---:|---:|---:|---:|---:|
| 0 | -0.0300 | -0.3030 | 0.1239 | -0.4672 |
| 1 | -0.5038 | -0.2001 | -0.1696 | -0.1440 |
| 2 | 0.0403 | -0.3430 | 0.0968 | -0.6202 |
| 3 | 0.6481 | 0.3660 | -0.0891 | -0.1018 |
| 4 | -0.2832 | -0.2471 | -0.3497 | 0.0408 |

Global seed 3 同时偏向大幅正 bid/ask，是最清晰的 catastrophic seed；state-dependent 组没有类似双侧 runaway。State-dependent ask side 仍有 seeds 0/2 的明显负 offset，所以 ask stability 只能称为小幅改善，不能声称 bad seeds 完全消失。

## 2. Long-run trajectory

Checkpoint-level cross-seed dispersion：

| Step | Global bid std | State bid std | Global ask std | State ask std |
|---:|---:|---:|---:|---:|
| 20,480 | 0.2947 | 0.1120 | 0.1394 | 0.0690 |
| 60,416 | 0.0889 | 0.2350 | 0.1285 | 0.0717 |
| 100,352 | 0.1715 | 0.2088 | 0.1683 | 0.1575 |
| 149,504 | 0.2152 | 0.1208 | 0.1791 | 0.2149 |
| 199,680 | 0.3162 | 0.1791 | 0.1865 | 0.2034 |
| 204,800 | 0.3747 | 0.1778 | 0.2475 | 0.2381 |

60k 后 global bid dispersion 持续扩大，主要来自 seed 1 的 late negative drift 和 seed 3 的 persistent positive runaway；state-dependent bid dispersion 到 204.8k 明显较低。Ask side 两组到 200k 都出现 drift，最终 dispersion 仅小幅不同。结论是：**persistent drift 在 bid side 明显减少，在 ask side没有消失。**

## 3. Economic outcome

| 5-seed mean | Global std | State-dependent std | 变化 |
|---|---:|---:|---:|
| Market share | 0.7128 | 0.8068 | +0.0940 |
| Spread PnL / step | 0.1855 | 0.2460 | +32.6% |
| Total PnL / step | 0.1078 | 0.2063 | +91.3% |
| Inventory std | 10.7631 | 8.5167 | -20.9% |

Paired-seed 结果也不是由单一 seed 驱动：spread PnL / step 与 total PnL / step 在 5/5 seeds 改善；market share 和 inventory std 在 4/5 seeds 改善。Market-share seed dispersion 也从 `0.0992` 降至 `0.0342`。

所以 mean-policy 更稳定的同时，经济表现没有付出明显代价，反而整体改善。

## 4. Persistent 环境中的 state-dependent variance

| Variance diagnostic（5-seed mean） | Global std | State-dependent std |
|---|---:|---:|
| Latent sigma bid mean | 0.9753 | 0.8262 |
| Latent sigma ask mean | 0.9728 | 0.7820 |
| `Std_s[σ_bid(s)]` | 0 | 0.3006 |
| `Std_s[σ_ask(s)]` | 0 | 0.4111 |

State-dependent 组所有 5 seeds 的 bid/ask sigma state std 都大于零：bid 范围 `0.0646–0.6168`，ask 范围 `0.0849–0.8292`。因此 network 在 Persistent 环境中仍然真实使用 state-dependent variance，而不是退化成近似 global constant。

## 对四个问题的直接回答

1. **是否降低 cross-seed dispersion？** 是。Bid 显著下降 54.8%，ask 小幅下降 5.6%，两侧平均下降 35.1%。
2. **60k+ persistent drift 是否减少？** 部分且明显偏向 bid side。Global 的 catastrophic bid drift 消失；ask drift 仍存在。
3. **PnL / market share 是否至少没有恶化？** 是，且四项 aggregate economic metrics 全部改善。
4. **σ(s) 是否仍有明显 state dependence？** 是。Bid/ask state dispersion 分别为 `0.3006/0.4111`，所有 seeds 均非零。

## Validation 与停止条件

- Core code 零修改：`src/ppo.py`、`src/networks.py`、`src/evaluation.py` 均未变。
- 10/10 existing tests、Python compile 和 global/state-dependent 双组 smoke 通过。
- 正式结果包含 2 conditions × 5 seeds × 204,800 steps，共 10 final rows；每组每 seed 8 个 checkpoints，共 80 trajectory rows。
- 未运行 competitor epsilon sweep、analytical optimum、sigma attribution、KL、value clipping、λ-return 或其他 PPO mechanism。
