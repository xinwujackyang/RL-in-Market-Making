# Correctness audit: timing, signs, and PPO

## Evidence and chosen event timeline

Ganesh et al. define inventory signs from the dealer's perspective: dealer buys are positive and dealer sells are negative. Their hedge is `-x_t z_t`, where `z_t` is known when the action is chosen. The adaptive-agent risk equation exposes `z_t(1-x_t) + v_t` to the next price move. The original project report is less precise, and the extracted notebooks instead generated investor flow, exposed the resulting inventory to the price move, and only then hedged a fraction of that unseen end-of-step inventory.

For a coherent decision process and the closest match to the paper, one step now means:

1. At time `t`, the dealer observes inventory `z_t`, price `P_t`, and previous-step PnL components.
2. The dealer chooses `(epsilon_bid, epsilon_ask, x_t)`.
3. The dealer hedges `-x_t z_t` at the exchange around `P_t`, leaving `z_t(1-x_t)` and paying a positive hedge cost.
4. Investors submit orders and choose the best dealer quote at `P_t`.
5. Dealer-perspective signed flow `v_t` is executed: buys are positive, sells are negative.
6. Inventory exposed to the next price move is `z_t(1-x_t) + v_t`.
7. The mid-price evolves from `P_t` to `P_{t+1}`.
8. Inventory PnL is `(P_{t+1}-P_t) [z_t(1-x_t)+v_t]`.
9. Reward and the next observation are produced.

The accounting identity is enforced as

`TotalPnL = SpreadPnL + InventoryPnL - HedgeCost`,

where `HedgeCost` is stored as a non-negative expense. This differs only in notation from sources that store the hedge cash flow as a negative PnL component.

## Sign examples

- Investor buys one unit: dealer uses its ask, sells one unit, inventory changes by `-1`, spread PnL is positive, and a subsequent price increase produces negative inventory PnL.
- Investor sells one unit: dealer uses its bid, buys one unit, inventory changes by `+1`, spread PnL is positive, and a subsequent price increase produces positive inventory PnL.
- Positive inventory hedge: the dealer sells at `P-S_ref`, inventory falls, and the positive cost is the sacrificed amount relative to mid-price.
- Negative inventory hedge: the dealer buys at `P+S_ref`, inventory rises toward zero, and the positive cost is again the amount paid away from mid-price.

Run `python experiments/diagnostics.py` for deterministic numerical versions of these examples.

## PPO audit

- Actions are now sampled in latent Gaussian space and smoothly transformed with `tanh`. Bid and ask map to `[-1,1]`; the hedge maps to `[0,1]` with `(1+tanh(u))/2`.
- Log probabilities include the `tanh` Jacobian and the hedge scale factor. PPO stores and reevaluates the exact transformed action executed by the simulator.
- GAE uses `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)` and reverse discounted accumulation with `gamma*lambda`.
- A rollout boundary is a truncation, not a terminal state, so the final value is bootstrapped.
- Critic returns use the same final bootstrap value.
- The PPO ratio is `exp(new_log_prob-old_log_prob)`, followed by the standard clipped surrogate objective.
- Advantages are normalized once when a rollout is consumed. Rewards, returns, and observations are not normalized or mutated.
- The reported entropy is explicitly the base-Gaussian entropy proxy. Exact transformed entropy is not used in the loss.

## Modeling differences from Ganesh et al.

The default baseline still uses Gamma-distributed order sizes, while the simulator now also supports paper-style unit orders through `order_size_mode="unit"`. It retains a deterministic size-based reference-spread curve, GBM with annualized parameters, a five-component partial observation, and exactly one competing dealer. Ganesh et al. use unit orders in core experiments, empirically calibrated stochastic reference spreads, richer observations, and RLlib PPO. Results should therefore be described as partial behavioral replication, not exact numerical replication.
