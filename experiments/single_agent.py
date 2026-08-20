from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import PersistentMarketMaker, RandomMarketMaker
from config import Config
from environments import SingleDealerMarketEnv
from evaluation import evaluate_baseline, evaluate_policy, print_metrics, save_plots
from ppo import PPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO in the single-dealer market")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--risk-penalty", action="store_true")
    args = parser.parse_args()
    cfg = Config(rollout=args.rollouts, horizon=args.horizon, eval_episodes=args.eval_episodes, eval_days=args.eval_days)
    steps = cfg.steps_per_day * cfg.eval_days
    env_factory = lambda episode: SingleDealerMarketEnv(cfg, seed=cfg.seed + 1000 + episode)

    agent = PPOAgent(SingleDealerMarketEnv(cfg), cfg)
    evaluator = lambda current: evaluate_policy(current, env_factory, cfg.eval_episodes, steps)[0]
    history = agent.train(evaluator=evaluator, risk_penalty=args.risk_penalty)
    metrics, data = evaluate_policy(agent, env_factory, cfg.eval_episodes, steps)
    print_metrics(metrics)
    print("\nRandom baseline:", evaluate_baseline(RandomMarketMaker(seed=cfg.seed), env_factory, cfg.eval_episodes, steps))
    print("Persistent baseline:", evaluate_baseline(PersistentMarketMaker(), env_factory, cfg.eval_episodes, steps))
    output = ROOT / "results" / "single_agent.png"
    output.parent.mkdir(exist_ok=True)
    save_plots(history, data, str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
