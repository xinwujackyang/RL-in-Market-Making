# RL Market Making: Final Research Summary

This report is the canonical English overview of the project. The corresponding
[Chinese research record](report_zh.md) is retained separately.

## 1. Research Question

> How does a PPO market maker learn quoting, inventory control, and hedging under
> competitive dealer markets, and how does it compare with heuristic, adaptive,
> and classical stochastic-control benchmarks?

The project began as a reproduction inspired by Ganesh et al. (2019), then evolved
into a controlled behavioral investigation. The research sequence was deliberate:
audit simulator and PPO correctness, stabilize the reference policy, analyze the
learned controls, and compare them with distinct economic benchmarks.

The project does not combine every controller into one PnL leaderboard. Persistent,
Adaptive, and A-S comparisons answer different questions under different matched
protocols.

## 2. Market Simulator

Two dealers compete for the same investor orders. At each 15-minute step, a dealer
chooses

```text
(epsilon_bid, epsilon_ask, hedge_fraction).
```

Smaller epsilon means a tighter, more aggressive quote relative to the simulator's
reference spread. The event sequence is:

1. hedge inventory known at decision time;
2. route investor orders;
3. update dealer-perspective inventory;
4. evolve the GBM mid-price;
5. realize PnL on post-hedge, post-flow inventory.

Accounting satisfies

```text
TotalPnL = SpreadPnL + InventoryPnL - HedgeCost.
```

Formal Adaptive and A-S benchmarks use 20 independent unit orders per step, balanced
buy/sell directions, and market volatility `sigma=0.2`. Independent RNG streams for
order size, direction, price, and routing support matched-path experiments.

Execution is winner-take-all: the better quote receives the entire investor order,
and an exact tie is split randomly. This is useful for controlled dealer competition,
but it is not a continuous limit-order-book fill model.

## 3. PPO Implementation and Correctness

The simulator and learner were audited before economic interpretation. The corrected
implementation includes:

- dealer-perspective inventory signs;
- explicit hedge → flow → price-move timing;
- consistent PnL identities;
- tanh-squashed bounded actions;
- Jacobian-corrected transformed log probabilities;
- identical sampled, executed, stored, and reevaluated PPO actions;
- GAE with rollout-boundary bootstrapping;
- standard PPO clipped policy ratios.

The final reference PPO uses a five-dimensional observation, relative price
`P_t/P_0 - 1`, separate `2 x 256` tanh actor and critic networks, and a
state-dependent squashed-Gaussian policy. Quote actions lie in `[-1, 1]` and the
hedge fraction lies in `[0, 1]`.

Two implementation findings materially improved the reference policy. Relative-price
scaling reduced first-layer actor saturation. State-dependent policy variance reduced
the Random-benchmark final deterministic two-side MAE from `0.1601` to `0.0808` and
seed dispersion from `0.1424` to `0.0467`. These results justify the reference
configuration; they are not claims of globally optimal PPO hyperparameters.

See the [correctness audit](../../docs/correctness_audit.md) for implementation-level
details.

## 4. Benchmark Taxonomy

| Method | Type | Learning mechanism | Inventory control | External hedge |
|---|---|---|---|---|
| Persistent | heuristic | none | none | no |
| Adaptive | empirical online learning | response table | quote skew | yes |
| A-S | stochastic control | none | analytical skew | no |
| PPO | model-free RL | neural policy | learned skew | yes |

The matched questions are:

| Research question | Comparison | Main result |
|---|---|---|
| Does PPO learn against a fixed dealer? | PPO vs Persistent | It learns nontrivial quoting and inventory-conditioned behavior. |
| Can PPO compete with an adaptive dealer? | PPO vs Adaptive | It eventually improves normalized margin and volume. |
| Does PPO rediscover classical inventory control? | PPO vs A-S | Yes in direction, not in exact functional form. |
| Does richer RL outperform A-S economically? | PPO vs A-S | No; A-S wins 2/3 seed paths. |

## 5. Learned PPO Behavior

The quote action is decomposed as

```text
m = (epsilon_bid + epsilon_ask) / 2
k = epsilon_bid - epsilon_ask,
```

where `m` is the symmetric quote level governing the overall margin-volume choice,
and `k` is the relative skew governing inventory liquidation.

Against the calibrated Adaptive dealer, the final inventory-skew coefficient is
positive in 3/3 seeds. PPO therefore learns an economically signed relative quote
response: long inventory makes the ask relatively more aggressive, while short
inventory makes the bid relatively more aggressive.

PPO also learns absolute hedge control. At the 100,352-step policy analysis,
`E[h|q|]` increases across inventory-magnitude bins as

```text
0.2816 -> 1.5599 -> 3.1131 -> 4.6862.
```

The hedge fraction itself does not increase monotonically, and residual exposure
still grows with inventory. The supported conclusion is absolute balance-sheet risk
reduction, not proportional or complete inventory neutralization.

## 6. PPO vs Adaptive Market Maker

The Adaptive dealer estimates quote-response statistics online and updates only the
cell corresponding to the executed quote. Under a learning opponent, this produced a
stale-response pathology: wide quotes received no flow, so alternative response cells
were not refreshed and the controller could remain near an absorbing state.

The stable benchmark adds deterministic diagonal probing every 20 adaptive steps.
This is a project-specific response-tracking mechanism, not a canonical component of
Ganesh et al. The frozen configuration uses a calibrated 11-by-11 response table,
target market share `0.5`, risk aversion `2`, retention factor `beta=0.35`, and a 5%
probe fraction.

Long-run PPO economics evolve as follows:

| Training step | PPO share | PPO total PnL/step | PPO normalized | Adaptive normalized | PPO/Adaptive ratio |
|---:|---:|---:|---:|---:|---:|
| 100,352 | 0.4560 | 0.1111 | 0.7450 | 0.6642 | 1.1218 |
| 150,528 | 0.4674 | 0.2541 | 0.9117 | 0.7597 | 1.2001 |
| 204,800 | 0.5416 | 0.4784 | 1.1460 | 0.8523 | 1.3447 |

At the final checkpoint, Adaptive total PnL/step is `0.2985`, and PPO has higher
total PnL in 3/3 seed paths. The mean PPO advantage `0.1799` decomposes into spread
`+0.1970`, inventory `-0.0170`, and hedge-cost difference `-0.0001`. The advantage
is therefore a spread-capture and margin-volume result, not better inventory PnL or
lower hedge cost.

There is no convergence claim. Share, quote level, spread per unit, and total PnL
continue to move at 204.8k steps, while PPO total-PnL population standard deviation
expands from `0.0332` at 100k to `0.3906` at 204.8k.

The normalization is

```text
(window spread PnL / captured volume) / window mean S_ref(1),
```

not an exact dealer-volume-weighted estimator, because the saved aggregate artifact
does not retain dealer-volume-weighted reference-spread exposure.

See the canonical [PPO vs Adaptive report](../ppo_vs_adaptive_probe20_longrun/report.md).

## 7. Avellaneda–Stoikov Benchmark

The A-S controller retains the canonical reservation-price and optimal-spread
structure:

```text
r = S - q gamma V
Delta = gamma V + 2/gamma log(1 + gamma/k).
```

The implemented benchmark is a normalized stationary compatibility model, not a
direct calibration of the original exponential Poisson-arrival model. It uses a
26-step one-trading-day receding risk horizon, local GBM variance, fixed normalized
`rho` and `kappa`, `inventory_anchor=20`, a zero-inventory quote aligned with the
unit reference quote, native winner-take-all execution, and zero external hedge.

No smooth execution elasticity is inferred from winner-take-all routing. In the
standalone 3-seed, 100,352-step sanity comparison, A-S has mean absolute inventory
`4.998`, versus `1131.933` for a zero-skew Persistent dealer. Conditional inventory
drifts have the correcting signs, clipping is 0%, and all PnL identities hold. The
Persistent comparison is an inventory-control sanity baseline, not a general claim
of A-S performance superiority.

See the canonical [A-S benchmark report](../as_benchmark/report.md).

## 8. PPO vs A-S

Before clipping, the analytical A-S skew is

```text
k_AS(q) = 0.1 q.
```

The final pooled deterministic PPO fit is

```text
k_PPO(q) = 0.1255 + 0.0476 q,    R^2 = 0.578.
```

For pooled states with `|q| >= 1`, PPO has the A-S corrective sign in `96.0%` of
states. All three PPO seeds have positive slopes, but per-seed corrective-sign
frequency ranges from `77.6%` to `100%`. The pooled slope is weaker than the A-S
slope and the binned policy visibly saturates. The regression is descriptive, not a
complete or causal model of the neural policy.

The precise conclusion is:

> PPO independently rediscovers the direction of A-S-like inventory control, but
> not the same control law.

Final matched deterministic economics are:

| Metric | PPO | A-S | PPO - A-S |
|---|---:|---:|---:|
| Total PnL/step | 0.1340 ± 0.0796 | 0.2388 ± 0.1252 | -0.1048 |
| Spread PnL/step | 0.1367 ± 0.0716 | 0.2322 ± 0.1289 | -0.0955 |
| Inventory PnL/step | 0.0131 ± 0.0058 | 0.0066 ± 0.0271 | +0.0065 |
| Hedge cost/step | 0.0158 ± 0.0040 | 0 | +0.0158 |

PPO wins 1/3 seed paths. The mean gap satisfies

```text
-0.1048 = -0.0955 + 0.0065 - 0.0158.
```

The gap is primarily lower spread capture plus PPO hedge cost, not adverse inventory
PnL. Under this matched final evaluation, A-S has higher mean total PnL; this does not
establish general superiority outside the specified simulator and protocol.

See the canonical [PPO vs A-S report](../ppo_vs_as/report.md).

## 9. Main Conclusions

1. **PPO learns economically meaningful controls.** The policy jointly adjusts the
   symmetric quote level, inventory-dependent relative skew, and absolute hedge
   quantity.
2. **Against Adaptive, PPO eventually improves both margin and volume.** At 204.8k
   steps it exceeds 50% share while preserving a normalized monetization advantage,
   but the experiment does not support a convergence claim.
3. **PPO independently rediscovers A-S-like inventory-control direction.** Its pooled
   sensitivity is weaker and its response is nonlinear and seed-dependent.
4. **The richer RL policy does not outperform A-S economically in the matched final
   experiment.** A-S wins 2/3 seed paths; the mean gap comes mainly from spread
   capture and PPO hedge cost.

More broadly, correct semantics, observation representation, and exploration
parameterization were more reliable than local KL, critic-loss, or short-horizon
training diagnostics. Multi-seed, long-horizon evidence remains necessary for
interpreting learned market-making behavior.

## 10. Limitations

- **Winner-take-all execution.** There are no probabilistic partial fills, queue
  positions, or continuous LOB dynamics.
- **A-S compatibility model.** The normalized stationary implementation is not an
  empirical calibration of the original Poisson-arrival model.
- **Adaptive probing.** Persistent diagonal probing is a project-specific tracking
  extension, not a canonical algorithmic component.
- **No convergence claim.** Adaptive-run economics have not plateaued by 204.8k and
  cross-seed dispersion is widening.
- **Simulation scope.** Synthetic GBM prices, one competitor at a time, a limited
  observation, and a narrow market configuration limit external validity.
- **Cross-benchmark comparability.** Results from Persistent, Adaptive, and A-S
  protocols should not be presented as one unified PnL ranking.

## 11. Future Research Directions

The benchmark scope is frozen. Concrete future research questions could examine:

1. probabilistic execution and partial fills;
2. calibration to real market microstructure;
3. opponent cross-play and out-of-distribution generalization;
4. an A-S extension with external hedging;
5. replay-based or real-data evaluation.

These are future directions, not open tasks required to interpret the current
project.
