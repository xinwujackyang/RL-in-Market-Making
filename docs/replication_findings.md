# Correctness and paper-replication findings

## Before/after baseline

The unchanged Gaussian-plus-clipping metrics and plots are in `results/baseline_unbounded/`. Corrected default-run outputs are in `results/corrected_default/`. The correction changes both the policy likelihood and the simulator's economic timing/signs, so PnL level changes should not be attributed to only one component.

The corrected stochastic five-seed policies put about 1% of epsilon actions within `0.01` of a bound and about 3% of hedge actions within `0.01` of a bound. Their base-Gaussian entropy proxy remains approximately `4.20-4.25`, hedge-fraction standard deviation is about `0.30`, and latent epsilon standard deviations remain close to `1.0` after 20 rollouts. There is no longer any mismatch between stored likelihood actions and executed actions.

## Analytical experiments

Against `Uniform[-1,1]`, Theorem 1 gives the deterministic analytical best response epsilon `0`. Across five seeds, sampled means are `0.073 +/- 0.135` bid and `0.129 +/- 0.185` ask, while within-policy sampled standard deviations are approximately `0.62`. This is near zero only after averaging unstable seeds; it is not reliable concentration.

Against persistent epsilon `0.5`, the deterministic spread-PnL supremum is approached at `0.5-`. Five-seed sampled means remain near zero with high variance. Removing price risk from training and using spread PnL as the reward does not move them toward `0.5-`, ruling out inventory and hedge terms as the main cause.

The persistent objective is a hard routing threshold. For a nonzero-variance policy, the relevant objective is

`E[(1+epsilon) 1{epsilon < 0.5}]`,

not the deterministic Theorem 1 objective. `experiments/analytical_response.py` computes its optimum for the tanh-Gaussian policy. At latent standard deviation near `1.0`, the optimal mean epsilon is slightly negative; at the long-run learned standard deviation near `0.72`, it is near zero. A 204,800-step spread-only run learns sampled means `(0.030, -0.009)` and market share `0.773`, closely matching that stochastic-policy explanation. The deterministic `0.5-` result emerges only as variance approaches zero.

## Inventory skew and risk aversion

The inventory-conditional aggregate does not show robust economic internalization. Economically correct behavior requires bid epsilon to rise with inventory and ask epsilon to fall with inventory. The random-competitor aggregate primarily shows the reverse; persistent curves are weak. The paper's quote-skew result is not reproduced.

The current-step inventory-PnL-squared penalty is the coherent Markov reward choice: it penalizes the risk realized by the action and flow in the same transition. At `alpha=0.01`, the three-seed paired comparison modestly reduces inventory standard deviation (`19.31` to `18.62`) and inventory-PnL standard deviation (`4.81` to `4.55`). This establishes a small risk effect but not a statistically reliable return tradeoff.

## Answers to the stop questions

1. **Bounded PPO consistency:** yes. Tanh transforms, hedge scaling, and Jacobians are included; stored and executed actions match.
2. **Economic signs:** yes. Dealer buys are positive, dealer sells negative, hedge costs positive expenses, and deterministic examples pass.
3. **Event timing:** yes. Known inventory is hedged, investor flow executes, post-hedge/post-flow inventory is exposed to `P_t -> P_{t+1}`, then reward is reported.
4. **Random analytical best response:** not reliably at the current budget. The five-seed average is near zero, but seed variation and within-policy dispersion are large.
5. **Persistent competitor:** the deterministic benchmark is missed because persistent routing is discontinuous and the learned stochastic policy retains material variance. The spread-only and long-run diagnostics support this explanation.
6. **Inventory-dependent skewing:** present only weakly and with the wrong aggregate economic sign; the paper result is not reproduced.
7. **Risk-return effect:** the penalty measurably but modestly reduces inventory risk; the three-seed sample does not establish a reliable return cost or benefit.
