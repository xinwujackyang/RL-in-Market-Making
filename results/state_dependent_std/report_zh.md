# State-Dependent Std 实验报告

## 结论

本实验属于 **Case A：mean-policy stability 明显改善，并且 σ(s) 确实表现出 state dependence**。

将全局三维 `log_std` 参数替换为零初始化的 state-dependent `policy_log_std(actor_hidden)` 后，Final two-side MAE 减半，seed dispersion 下降约三分之二，worst-side error 也接近减半。4/5 paired seeds 改善，且 60k+ aggregate trajectory 没有重现 baseline 的 persistent runaway。

改善并不是简单来自 exploration 全面收缩：final mean latent sigma 仍接近 1，sampled epsilon std 仍约为 `0.65–0.70`。关键变化是 sigma 被按 state 重新分配；final evaluation states 上的 sigma state dispersion 明显非零。因此 state-dependent exploration scale 是当前 custom PPO mean-policy stability 的重要改进，可作为新的 candidate baseline。下一步可单独研究 bounded-action distribution parameterization，本轮不自动继续。

## Controlled comparison

两组唯一 conceptual difference：

```text
Baseline: log σ 是所有 states 共享的三维参数
New:      log σ(s) = Linear(actor_hidden)，weight/bias 初始均为 0
```

因此新组初始时对所有 observations 都有 `σ(s)=1`，与 baseline 的初始 exploration scale 一致。

新组保持 five-feature relative-price observation、separate actor/critic、joint global gradient clipping、20 个 unit investors、Random `U[-1,1]` competitor、market sigma `0.2`、2×256 tanh、learning rate `5e-5`、clip `0.3`、gamma `0.999`、GAE λ `0.95`、10 epochs、minibatch 256、entropy coefficient `0.003`、value coefficient `0.5`、gradient clip `0.5`、bounded tanh-Gaussian、相同 5 seeds 和 204,800 steps。`gae_value_target=False`、`vf_clip_param=None`、`kl_coef=0`。Baseline 直接复用 `results/relative_price/`，没有重跑。

## Primary policy metrics

| 指标 | Global std baseline | State-dependent std | 变化 |
|---|---:|---:|---:|
| Final two-side MAE | 0.1601 | 0.0808 | -49.5% |
| Seed dispersion | 0.1424 | 0.0467 | -67.2% |
| Worst-side error | 0.4326 | 0.2250 | -48.0% |

Seed dispersion 沿用 baseline 口径：先计算每个 seed 的 bid/ask 平均绝对误差，再对 5 个 per-seed MAE 取 population standard deviation。

新组 per-seed final deterministic 结果：

| Seed | Bid ε | Ask ε | Two-side MAE | Baseline MAE |
|---:|---:|---:|---:|---:|
| 0 | 0.0083 | -0.0412 | 0.0247 | 0.1048 |
| 1 | -0.0215 | 0.0432 | 0.0324 | 0.0631 |
| 2 | 0.2250 | -0.0295 | 0.1272 | 0.1727 |
| 3 | 0.1677 | 0.1078 | 0.1377 | 0.4287 |
| 4 | 0.0362 | 0.1276 | 0.0819 | 0.0312 |

Seeds 0、1、2、3 改善；seed 4 变差，但绝对误差仍较小。Baseline catastrophic seed 3 从 MAE `0.4287` 降到 `0.1377`，新组没有 `|ε|≈0.4–0.9` 的 final bad seed。

## Mean latent sigma 与 state dependence

| Diagnostic（5-seed mean） | Global std baseline | State-dependent std |
|---|---:|---:|
| Latent sigma bid mean | 0.9797 | 0.9862 |
| Latent sigma ask mean | 0.9737 | 0.9166 |
| `Std_s[σ_bid(s)]` | 0 | 0.2592 |
| `Std_s[σ_ask(s)]` | 0 | 0.2714 |

Bid mean sigma 基本不变（约 +0.7%）；ask mean sigma 仅温和下降（约 -5.9%）。但 state dispersion 明显非零：

- Per-seed bid sigma state std 范围为 `0.0599–0.4681`。
- Per-seed ask sigma state std 范围为 `0.0852–0.6797`。
- 5 个 seeds 的 bid/ask state dispersion 都大于零，并非由单一 seed 驱动。

Trajectory 中 mean bid/ask sigma 合并平均从 10k 的 `0.991` 缓慢变到 204.8k 的 `0.940`；对应 sigma state dispersion 从 `0.063` 增到 `0.255`。Network 不仅具备 state-dependent capacity，PPO objective 也确实推动它使用了这一 capacity。

## Sampled policy variance

| Sampled epsilon std（5-seed mean） | Global std baseline | State-dependent std |
|---|---:|---:|
| Bid | 0.7509 | 0.6550 |
| Ask | 0.6850 | 0.6953 |

Bid sampled dispersion 下降约 12.8%，ask 反而小幅增加约 1.5%。两侧 sampled actions 仍然高度 stochastic，并没有收缩成近确定性 policy。

因此本轮支持的解释是：**mean-policy stability 的改善主要来自 variance 根据 state 重新分配，而不是 latent sigma 或 sampled action variance 的统一降低。**

## Long-run trajectory

| Step | Global std MAE | State-dependent std MAE | Global worst-side | State-dependent worst-side |
|---:|---:|---:|---:|---:|
| 20,480 | 0.1493 | 0.0759 | 0.6044 | 0.1801 |
| 60,416 | 0.1930 | 0.1153 | 0.5945 | 0.3828 |
| 100,352 | 0.1807 | 0.1309 | 0.6589 | 0.2982 |
| 149,504 | 0.1561 | 0.1151 | 0.3743 | 0.2309 |
| 199,680 | 0.1828 | 0.0621 | 0.4796 | 0.1930 |
| 204,800 | 0.1692 | 0.0695 | 0.4884 | 0.2110 |

60k–150k 期间仍可见 moderate oscillation，但没有形成持续恶化；200k 附近 aggregate MAE 反而进一步下降。各 seed 的 late offsets 保持在明显小于 baseline catastrophic tail 的范围内，说明 60k+ persistent runaway 显著减轻。

## 对计划问题的直接回答

1. **Final MAE 是否改善？** 是，`0.1601 → 0.0808`，下降 49.5%。
2. **Seed dispersion 是否改善？** 是，`0.1424 → 0.0467`，下降 67.2%。
3. **Worst-side / catastrophic drift 是否改善？** 是，`0.4326 → 0.2250`；未出现 final catastrophic seed。
4. **Mean latent sigma 是否变化？** Bid 基本不变；ask 温和下降。不是全面 variance collapse。
5. **σ(s) 是否真的表现出 state dependence？** 是，bid/ask 的 mean state std 为 `0.259/0.271`，所有 seeds 均非零。
6. **Sampled policy 仍然 stochastic 时，mean policy stability 是否改善？** 是。Sampled std 仍为 `0.655/0.695`，但 MAE 与 seed stability 均显著改善。

## Decision

将 state-dependent std 保留为新的 candidate baseline。配置 flag 默认仍为 `False`，以保持历史实验完全可复现；后续 candidate-baseline 实验应显式设置 `state_dependent_std=True`。

本轮不做 log-std clamp、variance regularization、std-specific learning rate、fixed-std comparison 或 optimizer 修改。

## Validation

- Initialization check：多个不同 observations 上初始 `σ(s)=1` 精确成立。
- Gradient check：两个不同 observations backward 后，`policy_log_std` weight 和 bias 均获得非零 gradient。
- `SquashedNormal`、Jacobian-corrected log-prob、entropy proxy、PPO ratio 和 rollout buffer 均未修改。
- 10/10 existing tests、Python compile、现有 diagnostics 和 short smoke run 通过。
- 正式输出包含 5 个 seeds、每 seed 204,800 steps，以及 8 个指定 checkpoints，共 40 条 trajectory rows。
