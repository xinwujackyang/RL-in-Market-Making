# RL Market Making

A compact research implementation of PPO market makers in a simulated dealer market, extracted from the original notebook project and audited for action-distribution, accounting, sign, and event-timing correctness. Original artifacts remain unchanged in `archive/`.

## Model

At each step a dealer observes its inventory, GBM mid-price, and previous PnL components, then chooses bid epsilon, ask epsilon, and a hedge fraction. Investor sizes are Gamma-distributed; each investor trades with the dealer offering the best price. In the two-dealer environment an RL dealer competes with either a random or persistent policy.

The event sequence is explicit: hedge inventory known at decision time, execute investor flow, evolve the mid-price, and realize inventory PnL on post-hedge/post-flow inventory. Dealer buys are positive inventory and dealer sells are negative. See `docs/correctness_audit.md` for the equations and hand-worked sign examples.

The custom PPO implementation includes GAE-lambda, clipped policy loss, value loss, entropy regularization, minibatches, and gradient clipping. Continuous actions use a tanh-squashed Gaussian with a Jacobian-corrected log probability; the hedge transform includes its `[0,1]` scale adjustment. The action stored by PPO is exactly the action executed by the simulator.

The complete audit conclusions and direct answers to the replication stop questions are in `docs/replication_findings.md`.

## Setup and core experiments

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python experiments/diagnostics.py
python experiments/single_agent.py
python experiments/two_agent_random.py
python experiments/two_agent_persistent.py
```

The defaults use 20 rollouts of 1,024 steps for iteration. Override `--rollouts`, `--horizon`, `--eval-episodes`, and `--eval-days` for longer studies.

## Correctness and replication studies

```bash
python experiments/replication_study.py       # five seeds by default
python experiments/analytical_response.py    # stochastic-policy best response
python experiments/risk_aversion.py           # paired three-seed comparison
python experiments/investor_flow_investigation.py  # Gamma vs paper unit flow
python experiments/paper_aligned_ppo.py             # unit-flow current vs paper PPO clip
python experiments/long_run_random.py               # 204,800-step convergence study
```

The investor-flow investigation keeps the current PPO fixed and compares the
default 20 Gamma-size orders with 20 paper-style unit orders under paired random
competitor seeds. Its Chinese report and latent-standard-deviation trajectory are
under `results/investor_flow/`. Run `--phase paper-ppo` only if the flow-only
comparison warrants the conditional full-paper configuration.

The paper-aligned PPO experiment reads the current defaults directly: both
conditions use learning rate `5e-5`, a two-layer 256-wide tanh network, and
minibatches of 256. The controlled comparison changes only PPO clip from `0.2`
to the paper-reported `0.3`, records deterministic mean-policy and latent-sigma
trajectories every rollout, and writes its five-seed report under
`results/paper_aligned_ppo/`.

The long-run Random study holds that paper-oriented setup fixed and changes
only the training budget from 20 to 200 rollouts. It uses sparse deterministic
evaluation checkpoints and writes five-seed convergence trajectories under
`results/long_run_random/`.

### Analytical best response

Ganesh et al. Theorem 1 gives epsilon `0` against `Uniform[-1,1]` and the supremum `0.5-` against a deterministic competitor quoting `0.5`.

At the current 20-rollout budget, five-seed random-competitor sampled means average `0.073` bid and `0.129` ask, but seed-to-seed variation is large and sampled epsilon standard deviations are about `0.61-0.62`. The policy is therefore near the analytical region on average but does **not** reliably concentrate there.

Against the persistent competitor, sampled means average `0.026` bid and `0.054` ask with about `0.62` standard deviation and `0.677` market share. A spread-only diagnostic gives `-0.031` bid, `-0.055` ask, and `0.735` market share. Extending one spread-only seed tenfold reduces latent standard deviation to about `0.72`, but still yields sampled means near zero and `0.773` market share.

This behavior is explained by the policy class and routing discontinuity. With a nonzero-variance tanh-Gaussian, maximizing expected spread PnL requires quoting well below `0.5` to avoid losing trades in the upper tail. The numerical response curve shows that at latent standard deviation `0.72` the stochastic optimum is near zero; `0.5-` is recovered only as variance approaches zero. PPO is therefore close to the stochastic-policy optimum in the spread-only case, even though it misses the deterministic theorem benchmark.

### Inventory-dependent skewing

The five-seed inventory-conditional plots do not reproduce robust economically signed skewing. The random-competitor aggregate tends to skew in the opposite direction, and the persistent/spread-only curves are weak. This is a negative replication result, not evidence of the paper's internalization behavior.

### Risk aversion

Using the current-step `-alpha * InventoryPnL^2` penalty with `alpha=0.01` reduces mean inventory standard deviation from `19.31` to `18.62` and inventory-PnL standard deviation from `4.81` to `4.55` across three paired seeds. Mean total PnL does not fall in this small sample (`110.83` versus `117.91`), but three seeds are insufficient to claim a return improvement. The supported conclusion is a modest risk reduction at the chosen coefficient.

## Known differences from Ganesh et al.

- Gamma-distributed order sizes remain the default; paper-style unit orders are available through `order_size_mode="unit"` and are used in the investor-flow A/B.
- A deterministic size-based reference-spread curve instead of a calibrated stochastic spread model.
- GBM parameters and scaling inherited from the project rather than the paper's exact setup.
- A five-component partial observation rather than the paper's richer trade-flow and market-share inputs.
- Custom PPO rather than RLlib PPO.
- Exactly one competitor; adaptive market makers are intentionally out of scope.

These are behavioral replication experiments, not an exact numerical reproduction.

## Known limitations

- Winner-take-all routing creates a discontinuous objective against persistent quotes.
- Policy variance remains high at the short default budget.
- Learned quote skew is not robust or correctly signed.
- Reference spreads and investor flow are deliberately simplified.
- Only random and persistent competitors are implemented.

## Repository layout

- `src/`: configuration, market mechanics, environments, baselines, network, PPO, and evaluation.
- `experiments/`: core runs plus correctness and replication diagnostics.
- `docs/correctness_audit.md`: timing, accounting, sign, and PPO audit.
- `notebooks/analysis.ipynb`: analysis-only notebook; no duplicate simulator or PPO code.
- `results/`: before/after baselines, multi-seed metrics, analytical response, and risk results.
- `archive/`: untouched original notebook, report, and paper.
