# Adaptive Market Maker: Decision Logic and Online Response Table

This module is a paper-inspired, explicitly documented interpretation of Ganesh et al.
(NeurIPS 2019). It implements deterministic decisions, online response updates,
a deterministic cold-start quote sequence, and minimal persistent probing.

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

An unpopulated cell raises `MissingResponseStatistics`. There are no default
values, priors, interpolation, or unseen-cell fallbacks.

## Online updates

The paper's `beta = 0.35` is interpreted as old-estimate retention. The first
observation at a cell initializes its mean and second moment directly:

```text
m = y
m2 = y**2
```

Later observations update both moments with the same EMA:

```text
m_new = 0.35 * m_old + 0.65 * y
m2_new = 0.35 * m2_old + 0.65 * y**2
```

Variance is recovered without bias correction as `max(m2 - m**2, 0)`. Only the
joint quote cell that was actually executed is updated; all other cells remain
unchanged.

These rules are implementation choices because the paper specifies only an
exponential forgetting factor, not the recursion, initialization, or variance
estimator.

## Cold start

`cold_start_quotes()` returns one deterministic full-grid pass of 121 unique
quotes. The first 11 quotes traverse the diagonal in ascending epsilon order.
The remaining off-diagonal cells use bid-major, then ask-major grid order.

This deterministic pass is an implementation choice: the paper does not define
a cold-start or exploration policy. The function only returns quotes; it is not
a scheduler, state machine, or adaptive exploration mechanism.

## Persistent probing

After cold start, `AdaptiveMarketMakerCompetitor` forces one symmetric diagonal
quote every 100 adaptive steps. The quote advances deterministically through
the epsilon grid in ascending round-robin order, beginning with `(-1, -1)`.
Probe quotes bypass Step 1 and Step 2 so that the executed response updates the
diagonal cell used by market-share targeting. The existing hedge objective is
still evaluated using that fixed diagonal quote and the current inventory.

The interval, diagonal-only support, round-robin order, and continued hedging
are implementation choices. Probing does not change the EMA, cold-start values,
or executed-cell-only update rule.

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
