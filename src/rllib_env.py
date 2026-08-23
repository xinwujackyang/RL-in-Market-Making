from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from baselines import RandomMarketMaker
from config import Config
from environments import TwoDealerMarketEnv


class RLlibMarketEnv(gym.Env):
    """Thin Gymnasium adapter around the existing two-dealer simulator."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any] | None = None) -> None:
        env_config = dict(env_config or {})
        self.cfg = Config(**env_config.get("market_config", {}))
        experiment_seed = int(env_config.get("seed", self.cfg.seed))
        episode = int(env_config.get("episode", 0))
        competitor = RandomMarketMaker(
            eps_bid_max=1.0,
            eps_ask_max=1.0,
            hedge_fraction_max=0.3,
            seed=experiment_seed * 100_000 + 20_000 + episode,
        )
        self.market = TwoDealerMarketEnv(
            competitor,
            self.cfg,
            seed=experiment_seed * 100_000 + 10_000 + episode,
            reward_mode="total",
        )
        observation = self.market.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return self.market.reset(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, _, info = self.market.step(np.asarray(action, dtype=np.float32))
        return observation, float(reward), False, False, info
