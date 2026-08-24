# Adaptive MM standalone validation

本实验是机制验证，不是正式 benchmark。Adaptive MM 先完成 121-step full-grid cold start，随后使用 online response table 对 PersistentMarketMaker(0.5, 0.5, 0) 报价和对冲。Investor size 使用现有 simulator 默认的 Gamma flow。

## 核心结果

| gamma | Market share | Mean abs inventory | Inventory std | Mean hedge | Short bid skew | Long ask skew | Spread PnL/step | Total PnL/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6606 | 15.8495 | 23.7704 | 0.0000 | 0.4531 | 0.4424 | 0.751345 | 0.785756 |
| 2 | 0.2898 | 2.9087 | 6.3254 | 0.2119 | 0.3730 | 0.3167 | 0.188517 | 0.146879 |

## 机制检查

- Cold start：所有 run 均为 121 steps，table 完整后连续进入 decide()。
- Market-share targeting：gamma=0 为 0.6606，gamma=2 为 0.2898；目标是 0.5。
- Short inventory：bid-side correct-skew frequency 为 0.8604 (gamma=0) / 0.4827 (gamma=2)。
- Long inventory：ask-side correct-skew frequency 为 0.8624 (gamma=0) / 0.4906 (gamma=2)。
- Risk aversion：mean hedge fraction 从 0.0000 变为 0.2119；mean |inventory| 从 15.8495 变为 2.9087。

## 发现的 implementation limitation

Market share 没有收敛到 0.5：gamma=0 overshoot，gamma=2 undershoot。Persistent quote 0.5 位于 0.2 grid 的 0.4 与 0.6 之间；deterministic routing 使 symmetric quote 的 realized share 接近 1 或 0，而不是 0.5。加上每个 cell 只有一次 cold-start observation，Step 1 的离散 response 无法稳定表达 50% target。

本轮不修改 grid、cold start 或 estimator，因此把这一点保留为后续需要明确设计的问题。它不是 simulator hook 的 flow/sign/timing 错误。

## 实现说明

Simulator 只增加 Adaptive-specific optional hooks；Random/Persistent 仍走原有 act(observation) 路径。本轮未训练 PPO，也未修改 response estimator、cold-start sequence 或 grid。
