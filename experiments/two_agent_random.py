from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import RandomMarketMaker
from config import Config
from environments import TwoDealerMarketEnv
from evaluation import evaluate_policy, print_metrics, save_plots
from ppo import PPOAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO against a random market maker")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=20)
    args = parser.parse_args()
    cfg = Config(rollout=args.rollouts, horizon=args.horizon, eval_episodes=args.eval_episodes, eval_days=args.eval_days, hidden_layers=3)

    def env_factory(episode: int):
        return TwoDealerMarketEnv(RandomMarketMaker(hedge_fraction_max=0.3, seed=cfg.seed + 2000 + episode), cfg, seed=cfg.seed + 1000 + episode)

    agent = PPOAgent(env_factory(0), cfg)
    steps = cfg.steps_per_day * cfg.eval_days
    evaluator = lambda current: evaluate_policy(current, env_factory, cfg.eval_episodes, steps)[0]
    history = agent.train(evaluator=evaluator)
    metrics, data = evaluate_policy(agent, env_factory, cfg.eval_episodes, steps)
    print_metrics(metrics)
    print("The deterministic analytical benchmark is epsilon=0; use replication_study.py for sampled five-seed results.")
    output = ROOT / "results" / "two_agent_random.png"
    output.parent.mkdir(exist_ok=True)
    save_plots(history, data, str(output), competitor=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
