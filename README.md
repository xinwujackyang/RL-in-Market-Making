# RL Market Making

A minimal research codebase extracted from the original notebook implementation of PPO market makers. The original artifacts are preserved unchanged in `archive/`; implementation lives in `src/`, executable studies in `experiments/`, and the notebook is analysis-only.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

```bash
python experiments/single_agent.py
python experiments/two_agent_random.py
python experiments/two_agent_persistent.py
```

Each command trains PPO, evaluates the deterministic policy, prints PnL and behavior statistics, and writes a four-panel plot under `results/`. Useful options are `--rollouts`, `--horizon`, `--eval-episodes`, and `--eval-days`. The single-agent experiment also supports `--risk-penalty`.

The defaults (20 rollouts of 1,024 steps) are intended for iteration. The original notebooks used 300 rollouts for the single-agent study and 100 rollouts of 2,048 steps for two-agent studies. For example:

```bash
python experiments/two_agent_random.py --rollouts 100 --horizon 2048
```

## Layout

- `src/config.py`: one dataclass containing market, PPO, and experiment settings.
- `src/market.py`: GBM, investor flow, spread, hedge, and PnL mechanics.
- `src/environments.py`: explicit single- and two-dealer environments.
- `src/baselines.py`: random and persistent policies with `act(observation)`.
- `src/networks.py` and `src/ppo.py`: custom actor-critic PPO with GAE, clipping, entropy, minibatches, and gradient clipping.
- `src/evaluation.py`: performance and policy-behavior evaluation plus plots.
- `notebooks/analysis.ipynb`: imports the modules and reads experiment outputs; it contains no simulator or PPO implementation.

## Replication notes

This pass preserves the notebooks' Gaussian-sampling-plus-clipping action approach. A TODO in `src/networks.py` marks it for the next PPO correctness pass. The original two-agent inventory sign arithmetic is also retained and documented in the environment instead of being silently changed. Persistent-competitor convergence may remain unstable, as reported in the original work.

No generic N-agent simulator, external RL library, experiment tracker, or configuration framework is included.
