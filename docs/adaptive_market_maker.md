# Adaptive Market Maker: Phase 1 Decision Logic

This module is a paper-faithful but documented interpretation of Ganesh et al.
(NeurIPS 2019). Phase 1 implements only deterministic decisions over an already
populated response table. It is not connected to either simulator, and it does
not define cold-start or online-learning behavior.

## Response table

The epsilon grid is `[-1.0, -0.8, ..., 0.8, 1.0]`. The table has one cell for
each joint `(epsilon_bid, epsilon_ask)` pair and performs no interpolation,
nearest-neighbor lookup, or continuous optimization.

Each populated cell stores:

- mean gross captured volume;
- mean and variance of signed net flow;
- mean and variance of normalized spread PnL.

Gross captured volume, rather than signed net flow, is used in Step 1. This is
an implementation choice that resolves an inconsistency in the paper: signed
flow cannot measure market share under balanced buy/sell flow. Signed net flow
continues to drive the Step 2 and hedge inventory-risk terms.

Normalized spread PnL is defined for future table updates as total timestep
spread PnL divided by the same timestep's `S_ref,t(0)`. It is not normalized by
volume or fill count.

An unpopulated cell raises `MissingResponseStatistics`. Phase 1 intentionally
has no default values, warm-up policy, or unseen-cell fallback.

## Decisions

Step 1 enumerates symmetric grid quotes and minimizes absolute target-share
error. It then selects the largest epsilon whose cost is within an absolute
`0.05` of the minimum cost.

Step 2 enumerates only correcting-side epsilon values no greater than the Step
1 epsilon. A short inventory changes only the bid; a long inventory changes
only the ask; zero inventory is not skewed. Equal objective values select the
largest epsilon, avoiding unnecessary quote aggression.

Hedging enumerates 101 fractions on `[0, 1]`. It uses the net-flow mean and
variance associated with the final post-skew quote. Equal objective values
select the smallest hedge fraction.

## Deferred online update choice

Online population is deliberately deferred. When implemented, the documented
choice is to interpret the paper's `beta = 0.35` as old-estimate retention:

```text
m_t = 0.35 * m_(t-1) + 0.65 * y_t
```

The mean and second moment will use the same EMA, with variance recovered as
`max(m2 - m1**2, 0)`. This rule is documented here but is not active in Phase 1.
