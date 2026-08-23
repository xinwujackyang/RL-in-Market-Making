# Modern RLlib PPO Reference Experiment

## 结论

在完全相同的 five-feature relative-price environment 上，现代 RLlib PPO **明显比 custom PPO 稳定**：

- 20k checkpoint 的 five-seed trajectory MAE 为 `0.1139`，能够在 early training 学到接近 `epsilon*=0` 的 policy；
- 60k 后存在 oscillation 和 transient drift，但没有 custom PPO 式的 persistent catastrophic runaway；
- Final two-side MAE：`0.1601 -> 0.1223`，下降 23.6%；
- Seed dispersion：`0.1424 -> 0.0577`，下降 59.5%；
- Worst-side error：`0.4326 -> 0.3073`，下降 29.0%；
- 五个 final seeds 均没有 `|epsilon| >= 0.4`。

因此结果属于预设的 Case A：

> **Custom PPO optimization/learner dynamics 是 remaining long-run instability 的主要嫌疑。RLlib 也有 transient drift，并未产生精确的 deterministic optimum，但其 drift 会回落，final seed stability 明显更好。**

这里比较的是完整 stack，而不是只替换某一行 update：RLlib 的默认 action-distribution parameterization、KL loss、value clipping、connector semantics 等与 custom global-log-std tanh-Gaussian 实现不同。因此证据定位到 learner/framework stack，但不能在本轮进一步归因到其中某一个机制。

## 实验控制

两组使用同一 current-best environment：

```text
observation = [inventory, relative price, TotalPnL, inventory PnL, HedgeCost]
20 unit-size investors
Random competitor U[-1,1]
buy probability = 0.5
mu = 0
sigma = 0.2
same routing, hedge, reward, PnL, and event timing
```

RLlib 使用一个薄 Gymnasium wrapper，内部直接调用现有 `TwoDealerMarketEnv`，没有复制 market logic 或创建第二套 simulator。Training 使用一个 local env runner、一个 local learner、无 distributed workers、无 Ray Tune。

能够直接对应的参数已匹配：

```text
framework = torch
separate actor/critic MLPs = 2 x 256 tanh
lr = 5e-5
clip = 0.3
gamma = 0.999
GAE lambda = 0.95
train batch = 1024
minibatch = 256
epochs = 10
entropy coefficient = 0.003
value coefficient = 0.5
gradient clipping = 0.5
total steps = 204,800
```

Custom PPO baseline 没有重跑，直接复用已有结果。

## 运行版本

| Component | Version |
|---|---:|
| Python | 3.11.15 |
| Ray | 2.57.0 |
| RLlib | bundled with Ray 2.57.0 |
| PyTorch | 2.12.0 |
| Gymnasium | 1.2.2 |

这是 **modern RLlib PPO reference**，不是 historical 2019 RLlib 或 Ganesh et al. exact reproduction。

## 保留的 RLlib defaults

没有调节 RLlib-specific 参数。实际默认值包括：

```text
new RLModule/Learner API stack = enabled
default continuous distribution = TorchDiagGaussian
free_log_std = False
KL loss = enabled
KL coefficient = 0.2
KL target = 0.01
value clipping = 10.0
shuffle batch per epoch = True
observation filter = NoFilter
normalize actions = True
clip actions = False
```

RLlib 默认 Gaussian action 通过标准 module-to-env connector unsquash 到环境 bounds；evaluation 复用同一默认 distribution class 与 RLlib `unsquash_action` 语义，没有实现 custom distribution、RLModule 或 PPO loss。

## 1. RLlib early training 能否找到 epsilon≈0？

**可以。** 20,480 steps 的 aggregate two-side MAE 为 `0.1139`，seed dispersion 为 `0.0514`，该 checkpoint 的 worst-side error 为 `0.2111`。

| Seed | 20k bid | 20k ask | Two-side MAE |
|---:|---:|---:|---:|
| 0 | 0.057 | 0.137 | 0.097 |
| 1 | -0.185 | -0.211 | 0.198 |
| 2 | 0.066 | 0.011 | 0.038 |
| 3 | -0.026 | -0.198 | 0.112 |
| 4 | -0.117 | -0.131 | 0.124 |

相同 checkpoint 的 custom PPO trajectory MAE 为 `0.1493`，dispersion 为 `0.1598`。RLlib early mean accuracy 和 cross-seed consistency 都更好。

## 2. 60k 后是否 drift？

**有 transient drift/oscillation，但没有 aggregate persistent runaway。**

| Step | RLlib MAE | RLlib dispersion | Worst-side error |
|---:|---:|---:|---:|
| 20,480 | 0.1139 | 0.0514 | 0.2111 |
| 60,416 | 0.1311 | 0.0668 | 0.2903 |
| 100,352 | 0.0820 | 0.0686 | 0.3916 |
| 149,504 | 0.1604 | 0.1340 | 0.4540 |
| 199,680 | 0.1047 | 0.0480 | 0.2342 |
| 204,800 | 0.1035 | 0.0562 | 0.2444 |

Seeds 3 和 4 在 150k 附近有明显 transient drift：seed 3 bid 达到约 `0.45`，seed 4 bid 达到约 `0.42`。但两者随后回落；到 200k aggregate MAE 和 dispersion 都重新下降。Seed 0 ask 在后期形成约 `-0.15` 的 moderate offset，seed 2 ask 约 `0.16`，但没有继续向 action bound runaway。

因此不能说 RLlib policy 完全无 drift，但关键差别是 drift 没有演化为 persistent catastrophic seed。

## 3. Final accuracy / seed stability

Final metrics 使用相同的独立 10-episode、20-day deterministic evaluation setup。

| Metric | Custom PPO | RLlib PPO | Change |
|---|---:|---:|---:|
| Final bid MAE | 0.1512 | 0.1112 | -26.5% |
| Final ask MAE | 0.1690 | 0.1334 | -21.1% |
| Final two-side MAE | 0.1601 | 0.1223 | -23.6% |
| Seed dispersion | 0.1424 | 0.0577 | -59.5% |
| Worst-side error | 0.4326 | 0.3073 | -29.0% |

最大的改善是 seed dispersion，而不是 mean accuracy。RLlib final MAE 仍非接近 0 的 replication closure，但 bad-seed tail 明显收缩。

## 4. Catastrophic seed 是否仍存在？

Final deterministic policies：

| Seed | Bid mean | Ask mean | Two-side MAE |
|---:|---:|---:|---:|
| 0 | 0.0455 | -0.1742 | 0.1098 |
| 1 | 0.1057 | 0.0216 | 0.0637 |
| 2 | -0.0168 | 0.1425 | 0.0797 |
| 3 | 0.2394 | 0.0214 | 0.1304 |
| 4 | 0.1484 | 0.3073 | 0.2279 |

**没有 final catastrophic seed。** Worst side 是 seed 4 ask `0.3073`，低于计划关注的 `|epsilon| >= 0.4`，也明显低于此前 custom/Paper-observation runs 中约 `0.8-0.9` 的 failures。

## Policy stochasticity

RLlib stochastic evaluation 的 sampled epsilon std：

```text
bid = 0.7413 +/- 0.0612
ask = 0.7285 +/- 0.0844
```

这没有显示 policy 已收缩为接近 deterministic 的 best response。由于 RLlib 使用默认 state-dependent diagonal Gaussian，而 custom PPO 使用 global-log-std tanh-Gaussian，不能把内部 latent std 做 apples-to-apples 比较。本轮不进一步研究 variance mechanism。

## 最终判断

四个问题的直接回答：

1. **Early learning：可以，20k MAE 为 0.1139。**
2. **60k+ drift：存在 transient drift，但没有 persistent runaway，200k 时重新回落。**
3. **Final accuracy/stability：RLlib MAE 改善 23.6%，seed dispersion 改善 59.5%。**
4. **Catastrophic seed：final 不存在，worst-side 为 0.3073。**

因此 custom PPO learner/framework dynamics 是当前 instability 的主要嫌疑。下一步若继续，应比较 update semantics、value loss、action distribution、advantage handling 和 epoch/minibatch behavior；本轮按停止条件不展开这些机制。

## Validation

- Adapter observation dimension：5；
- Adapter action dimension：3；
- 同 seed、同 fixed actions 下，wrapper 与原 environment 的 reward、next observation 和完整 info 连续多步完全一致；
- 2-batch RLlib smoke run 成功，准确累计 2,048 env steps；
- Deterministic/stochastic evaluation 正常；
- 所有送入 environment 的 RLlib actions 均通过原始 action-space bounds 检查；
- Existing tests：10/10 通过；
- Python compile：通过。
