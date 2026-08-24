# Calibrated / warm-start Adaptive MM screening

## Setup

3 seeds（0–2）× 100,352 formal PPO training steps。每 seed 先初始化唯一的 PPOAgent θ0，在独立 market path 上冻结参数并执行 10 × 121 = 1,210 calibration steps；随后恢复 post-initialization RNG state，将同一个 PPOAgent 接到 fresh formal simulator。正式训练从 populated 121-cell response table 开始，cold start=0，并继续 interval=100 diagonal probing。Fresh state 的 lagged market-volume denominator 初始化为 unit-flow 下确定的 20；price/inventory/PnL/path 均不从 calibration 继承。所有其他 PPO/environment/Adaptive 参数与 Phase 4b 相同。

## Adaptive: Phase 4b cold start vs calibrated

| Step | Cold share | Calibrated share | Cold mean base | Calibrated mean base | Cold base=1 | Calibrated base=1 |
|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.4140 | 0.3722 | 0.452 | 0.511 | 31.134% | 39.173% |
| 60,416 | 0.2895 | 0.3273 | 0.662 | 0.670 | 53.881% | 48.107% |
| 100,352 | 0.3168 | 0.2710 | 0.696 | 0.695 | 49.243% | 56.638% |

## PPO trajectory

| Step | Share | Spread PnL/step | Total PnL/step | Mean abs inventory | Inventory std | Mean hedge | Mean bid/ask epsilon |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.6278 | 0.2357 | 0.1709 | 5.587 | 7.041 | 0.473 | 0.105 / 0.025 |
| 60,416 | 0.6727 | 0.1962 | 0.1642 | 6.574 | 7.725 | 0.355 | 0.258 / 0.153 |
| 100,352 | 0.7290 | 0.3836 | 0.3201 | 7.112 | 7.867 | 0.312 | 0.304 / 0.207 |

## Adaptive trajectory

| Step | Share | Mean abs inventory | Inventory std | Mean hedge | Mean base | Mean bid/ask epsilon | Base=1 frequency | Probe fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.3722 | 3.657 | 5.767 | 0.209 | 0.511 | 0.315 / 0.276 | 39.173% | 0.996% |
| 60,416 | 0.3273 | 3.053 | 5.027 | 0.171 | 0.670 | 0.463 / 0.423 | 48.107% | 1.001% |
| 100,352 | 0.2710 | 2.498 | 4.385 | 0.126 | 0.695 | 0.518 / 0.509 | 56.638% | 1.001% |

## Final by seed

| Seed | PPO total PnL/step | PPO share | PPO mean abs inventory | Adaptive share | Adaptive mean base | Base=1 frequency |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2298 | 0.7756 | 5.412 | 0.2244 | 0.811 | 64.962% |
| 1 | 0.5652 | 0.8856 | 9.637 | 0.1144 | 0.937 | 85.983% |
| 2 | 0.1651 | 0.5258 | 6.287 | 0.4742 | 0.338 | 18.969% |

## Final dispersion: cold start vs calibrated

| Metric | Cold-start std | Calibrated std |
|---|---:|---:|
| Adaptive share | 0.0927 | 0.1505 |
| Adaptive base=1 frequency | 17.138% | 27.984% |
| PPO total PnL/step | 0.0258 | 0.1753 |

## PPO diagnostics

| Step | Approx KL | Clip fraction | Value loss |
|---:|---:|---:|---:|
| 20,480 | 0.0051 | 0.0254 | 1949.6 |
| 60,416 | 0.0059 | 0.0271 | 6036.7 |
| 100,352 | 0.0009 | 0.0017 | 39334.9 |

## Interpretation

- **Calibration/lifecycle 按设计完成。** 每 seed 使用同一个 θ0 PPOAgent；1,210 calibration steps 覆盖 10 次完整 121-cell grid，parameter max change=0、optimizer 未更新，并恢复 post-initialization RNG state。Formal simulator 为 fresh P0/zero-inventory state，response table 完整，cold start=0，calibration steps 不计入100,352 training steps。
- **Warm start 没有提高 Adaptive final stability。** Final mean share 从 cold-start Phase 4b 的 0.3168 降至 0.2710；mean base=1 frequency 从 49.2% 升至 56.6%。
- **三条 seed 分化而非共同稳定。** seed 0: share 0.392→0.407→0.224, base=1 37.1%→38.0%→65.0%；seed 1: share 0.351→0.127→0.114, base=1 41.8%→82.8%→86.0%；seed 2: share 0.374→0.448→0.474, base=1 38.5%→23.6%→19.0%。Seed 1 快速进入 PPO-dominant / near-lock-in tendency，seed 2 则向更平衡 interaction 移动，seed 0 介于两者之间。100k 尚未达到 share≈0/base=1≈100%，但趋势不满足 3/3 healthy criterion。
- **Seed dispersion 全面恶化。** Adaptive share std 0.0927→0.1505；base=1 std 17.1%→28.0%；PPO PnL std 0.0258→0.1753。Warm start 没有降低initial-condition/path sensitivity。
- **PPO mean economics 改善但不跨 seed 一致。** Total PnL/step 0.1709→0.3201，2/3 seeds 改善；market share 0.6278→0.7290。Final PnL mean 高于 cold-start 0.2838，但主要由 seed 1 的 PPO-dominant path 拉高，不能作为 benchmark 成功证据。
- **Inventory exposure 更高。** PPO mean abs inventory 5.587→7.112；warm-start final 高于 cold-start 6.324。Seed 1 final 达 9.637。
- Final value loss range=1905.6–107960.9，最大值同样来自 seed 1；KL/clip 未共同爆炸。
- Execution sanity 全部通过：3/3 calibration 与 formal runs 完成；每 seed cells=121、formal cold start=0、formal probes=1,003；无 NaN/Inf、越界 action、parameter drift 或 crash。

## Go / No-Go

**No-Go for 5-seed × 204,800 warm-start confirmation.** 结果不支持 H1（instability 主要来自 immature initial beliefs）；更一致的是 H2：即使以 θ0 下的 calibrated table 开始，online learning interaction 仍可快速分化。Warm-start Adaptive 不应取代 Phase 4b setup 成为主要 benchmark candidate。下一步若继续，应诊断 diagonal tracking、off-diagonal staleness 与 Step 1/Step 2 feedback，而不是扩大本配置样本量。
