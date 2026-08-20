from __future__ import annotations

import numpy as np


class RandomMarketMaker:
    def __init__(
        self,
        eps_bid_max: float = 1.0,
        eps_ask_max: float = 1.0,
        hedge_fraction_max: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.eps_bid_max = float(np.clip(eps_bid_max, 0.0, 1.0))
        self.eps_ask_max = float(np.clip(eps_ask_max, 0.0, 1.0))
        self.hedge_fraction_max = float(np.clip(hedge_fraction_max, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray | None = None) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(-self.eps_bid_max, self.eps_bid_max),
                self.rng.uniform(-self.eps_ask_max, self.eps_ask_max),
                self.rng.uniform(0.0, self.hedge_fraction_max),
            ],
            dtype=np.float32,
        )


class PersistentMarketMaker:
    def __init__(self, epsilon_bid: float = 0.5, epsilon_ask: float = 0.5, hedge_fraction: float = 0.0) -> None:
        self.action = np.array(
            [
                np.clip(epsilon_bid, -1.0, 1.0),
                np.clip(epsilon_ask, -1.0, 1.0),
                np.clip(hedge_fraction, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def act(self, observation: np.ndarray | None = None) -> np.ndarray:
        return self.action.copy()
