# Normalized Stationary Avellaneda–Stoikov Benchmark

This report defines the canonical English interpretation of the A-S benchmark used
in this repository. The [Chinese research record](report_zh.md) remains available.

## Motivation

The simulator routes each investor order to the dealer offering the better quote.
Market share is therefore approximately a step function of relative price, not a
smooth response from which an execution elasticity can be identified.

The benchmark is consequently defined as:

> an analytical A-S quote controller with execution, inventory updates, price
> evolution, and PnL supplied by the native simulator.

No environment is created specifically for A-S, and no smooth arrival parameter is
fitted from winner-take-all routing.

## Canonical A-S Structure

The controller retains the canonical reservation-price structure

```text
r = S - q gamma V,
```

where `V` is the conditional price variance over a fixed risk horizon. The total
spread is

```text
Delta = gamma V + 2/gamma log(1 + gamma/k).
```

The corresponding distances from the mid-price are

```text
delta_bid =  q gamma V + Delta/2
delta_ask = -q gamma V + Delta/2.
```

Positive inventory therefore makes the bid more passive and the ask more aggressive;
negative inventory reverses the skew.

## Compatibility With This Simulator

The implementation deliberately differs from a textbook finite-horizon A-S model in
the following ways:

- the risk horizon is a fixed, receding one-trading-day horizon rather than time to
  the end of PPO training;
- dollar parameters are scale-normalized by the prevailing unit-flow reference quote;
- the neutral quote is aligned with the simulator's canonical reference quote;
- execution uses native winner-take-all routing rather than exponential Poisson
  arrival intensities;
- the dealer uses quote skew only and has no external hedge;
- executable quote actions are clipped to the simulator domain, with raw actions and
  clipping recorded explicitly.

This is a **normalized stationary compatibility implementation**, not a direct
calibration of the original A-S arrival model.

## Parameterization

For a unit order, the reference quote distance is

```text
D_t = S_ref,t(1) = 2.2e-4 P_t.
```

The local Brownian-price variance over `H` steps is

```text
V_t(H) = P_t^2 sigma^2 dt H,
```

with `H=26`, corresponding to one 6.5-hour trading day at 15 minutes per step.

Define normalized coordinates

```text
nu = V / D^2
rho = gamma D
kappa = k D.
```

The normalized neutral distance and inventory-skewed quote distances are

```text
d_0 = rho nu / 2 + log(1 + rho/kappa) / rho
d_bid = d_0 + q rho nu
d_ask = d_0 - q rho nu.
```

Simulator actions are `epsilon_bid=d_bid-1` and `epsilon_ask=d_ask-1`.

Two anchors determine the frozen normalized parameters:

1. at zero inventory, `epsilon_bid=epsilon_ask=0`, so `d_0=1`;
2. `inventory_anchor=20` produces one full reference-distance inventory shift.

The second anchor uses the market's 20 units of investor flow per step, not PPO
results or PnL tuning. It implies

```text
rho nu = 1/20 = 0.05
k(q) = epsilon_bid - epsilon_ask = 0.1 q
```

before clipping. The resulting frozen values are

```text
rho   = 1.5246e-05
kappa = 1.02563340266.
```

At `P_0=100`, the dollar-coordinate equivalents are `gamma=0.000693` and
`k=46.6197001209`. These dollar equivalents vary with the prevailing reference
spread; `rho` and `kappa` remain fixed.

## Standalone Validation

A-S was run against a Persistent dealer quoting

```text
epsilon_bid = epsilon_ask = 0
hedge_fraction = 0
```

for 3 seeds and 100,352 steps per seed under unit flow and `sigma=0.2`. Both dealers
used the same `TwoDealerMarketEnv`, investor orders, and price path.

| Metric | A-S | Persistent |
|---|---:|---:|
| Market share | 0.4998 ± 0.0006 | 0.5002 ± 0.0006 |
| Spread PnL/step | 0.1686 ± 0.0655 | 0.2250 ± 0.0875 |
| Normalized spread monetization | 0.7499 ± 0.0002 | 1.0000 ± 0.0000 |
| Total PnL/step | 0.1721 ± 0.0669 | -0.1028 ± 0.3813 |
| Mean absolute inventory | **4.998 ± 0.006** | 1131.933 ± 864.701 |
| Inventory standard deviation | **5.912 ± 0.007** | 621.310 ± 370.218 |

Values are three-seed means with population standard deviations.

The Persistent dealer is an inventory-control sanity baseline. Its unbounded random
walk is not presented as a broad economic performance benchmark or as evidence that
A-S would dominate realistic dealer strategies.

## Inventory Mean Reversion

The most direct validation is conditional inventory drift:

```text
E[Delta q | q > 0] = -9.9962 ± 0.0142
E[Delta q | q < 0] =  10.0010 ± 0.0208.
```

The signs are economically correct: long inventory produces net selling flow and
short inventory produces net buying flow. Bid, ask, and pooled side clipping
frequencies are all `0.0%` in the standalone run. Every step and aggregate satisfy
the simulator PnL identity, and all recorded statistics are finite.

These checks validate that the analytical skew liquidates inventory under native
execution. They do not calibrate `k` from order-arrival data.

## Limitations

- Winner-take-all routing is discontinuous and lacks partial fills or queue position.
- The local GBM variance is an approximation, not an estimated intraday risk model.
- `inventory_anchor=20` is an interpretable simulator-scale anchor, not an empirical
  risk-preference estimate.
- Fixed normalized parameters imply price-varying dollar-coordinate `gamma` and `k`.
- The controller has no external hedge; an A-S-plus-hedge policy would be a separate
  benchmark.
- The standalone Persistent comparison validates inventory control but does not
  establish general economic superiority.
