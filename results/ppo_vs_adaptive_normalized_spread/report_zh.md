# Phase 12b：Normalized Spread Monetization Sanity Check

## 定义与数据边界

本分析无需重新训练。使用 simulator-native unit-size reference spread `S_ref,t(1) = reference_spread(P_t, 1, cfg)`；formal environment 的 price RNG 独立于 order/routing RNG，因此可用原 seed 精确重建每一步的 reference-spread path。当前参数下 `S_ref,t(1) = (2.0 + 0.2) x 1e-4 x P_t = 2.2e-4 P_t`。

Phase 12 artifact 没有保存逐步 dealer captured volume，因此不能无 rerun 地精确恢复 `sum_t V_dealer,t S_ref,t(1)`。本轮固定使用唯一的 market-scale normalization：`normalized = (window aggregate spread PnL / captured volume) / window mean S_ref,t(1)`。在 unit flow 下，每一步全部 20 个 market units 共享该 `S_ref,t(1)`；但该结果不冒充 dealer-volume-weighted estimator。

## 3-seed trajectory：mean（population std）

| Step | Mean S_ref(1) | PPO raw/unit | Adaptive raw/unit | PPO normalized | Adaptive normalized | PPO/Adaptive ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 20,480 | 0.022119 (0.002407) | 0.0132 (0.0016) | 0.0131 (0.0020) | 0.5970 (0.0155) | 0.5910 (0.0307) | 1.0103 |
| 60,416 | 0.015238 (0.003976) | 0.0097 (0.0028) | 0.0098 (0.0030) | 0.6290 (0.0539) | 0.6399 (0.0260) | 0.9831 |
| 100,352 | 0.026849 (0.012440) | 0.0192 (0.0079) | 0.0178 (0.0085) | 0.7450 (0.0903) | 0.6642 (0.0492) | 1.1218 |
| 150,528 | 0.042699 (0.022273) | 0.0373 (0.0191) | 0.0305 (0.0154) | 0.9117 (0.1115) | 0.7597 (0.1218) | 1.2001 |
| 204,800 | 0.050542 (0.032466) | 0.0554 (0.0413) | 0.0411 (0.0293) | 1.1460 (0.2551) | 0.8523 (0.1613) | 1.3447 |

## Final 204,800：按 seed

| Seed | PPO normalized | Adaptive normalized | Ratio |
|---:|---:|---:|---:|
| 0 | 1.3645 | 1.0058 | 1.3566 |
| 1 | 0.7882 | 0.6294 | 1.2522 |
| 2 | 1.2854 | 0.9215 | 1.3949 |

## 结论

**Case A。** PPO/Adaptive normalized ratio 从 100k 的 1.1218 升至 150k 的 1.2001 和 200k 的 1.3447；final PPO 在 3/3 seeds 上均高于 Adaptive。Phase 12 的 PPO per-unit advantage 在 simulator-native market spread scale normalization 后仍成立且扩大，因此后期 raw spread/unit 上升不能仅由 nominal price/reference-spread scaling 解释。

按照 stop rule，本分析到此结束，不继续 PPO-vs-Adaptive diagnostics。
