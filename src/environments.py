from __future__ import annotations

import numpy as np

from config import Config
from market import PnL, evolve_mid_price, execute_hedge, generate_investor_orders, reference_spread


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.array(
        [np.clip(action[0], -1, 1), np.clip(action[1], -1, 1), np.clip(action[2], 0, 1)],
        dtype=np.float32,
    )


class SingleDealerMarketEnv:
    """Single dealer environment extracted from MMRL_1MM.ipynb."""

    def __init__(self, cfg: Config | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.price = self.cfg.P0
        self.inventory = 0.0
        self.last_pnl = PnL()
        self.t = 0
        return self._observation()

    def _observation(self) -> np.ndarray:
        return np.array(
            [self.inventory, self.price, self.last_pnl.spread, self.last_pnl.inventory, -self.last_pnl.hedge_cost],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        eps_bid, eps_ask, hedge_fraction = _clip_action(action)
        spread_pnl = 0.0
        for size, direction in generate_investor_orders(self.cfg, self.rng):
            epsilon = eps_bid if direction == 1 else eps_ask
            spread_pnl += size * reference_spread(self.price, size, self.cfg) * (1 + epsilon)
            self.inventory += direction * size

        old_price = self.price
        self.price = evolve_mid_price(self.price, self.cfg, self.rng)
        inventory_pnl = (self.price - old_price) * self.inventory
        self.inventory, hedge_cost = execute_hedge(self.inventory, hedge_fraction, self.price, self.cfg)
        self.last_pnl = PnL(spread_pnl, inventory_pnl, hedge_cost)
        self.t += 1
        info = self._info(self.last_pnl)
        return self._observation(), self.last_pnl.total, False, info

    def _info(self, pnl: PnL) -> dict:
        return {
            "spread_pnl": pnl.spread,
            "inventory_pnl": pnl.inventory,
            "hedge_cost": pnl.hedge_cost,
            "total_pnl": pnl.total,
            "inventory": self.inventory,
        }


class TwoDealerMarketEnv:
    """RL dealer (index 0) competes with one supplied baseline dealer."""

    def __init__(self, competitor, cfg: Config | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or Config()
        self.competitor = competitor
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.price = self.cfg.P0
        self.inventory = [0.0, 0.0]
        self.last_pnl = [PnL(), PnL()]
        self.market_share = [0, 0]
        self.t = 0
        return self._observation(0)

    def _observation(self, dealer: int) -> np.ndarray:
        pnl = self.last_pnl[dealer]
        return np.array([self.inventory[dealer], self.price, pnl.total, pnl.inventory, pnl.hedge_cost], dtype=np.float32)

    def step(self, rl_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        actions = [_clip_action(rl_action), _clip_action(self.competitor.act(self._observation(1)))]
        spread_pnl = [0.0, 0.0]
        step_fills = [0, 0]

        for size, direction in generate_investor_orders(self.cfg, self.rng):
            ref = reference_spread(self.price, size, self.cfg)
            # Notebook convention: +1 selects asks; -1 selects bids. Inventory
            # arithmetic is intentionally preserved for the replication pass.
            prices = []
            for eps_bid, eps_ask, _ in actions:
                prices.append(self.price + ref * (1 + eps_ask) if direction == 1 else self.price - ref * (1 + eps_bid))
            if prices[0] == prices[1]:
                winner = int(self.rng.random() >= 0.5)
            elif direction == 1:
                winner = int(prices[1] < prices[0])
            else:
                winner = int(prices[1] > prices[0])
            epsilon = actions[winner][1 if direction == 1 else 0]
            spread_pnl[winner] += size * ref * (1 + epsilon)
            self.inventory[winner] += direction * size
            step_fills[winner] += 1

        old_price = self.price
        self.price = evolve_mid_price(self.price, self.cfg, self.rng)
        pnls = []
        for dealer, action in enumerate(actions):
            inv_pnl = (self.price - old_price) * self.inventory[dealer]
            self.inventory[dealer], hedge_cost = execute_hedge(self.inventory[dealer], action[2], self.price, self.cfg)
            pnls.append(PnL(spread_pnl[dealer], inv_pnl, hedge_cost))
            self.market_share[dealer] += step_fills[dealer]
        self.last_pnl = pnls
        self.t += 1
        total_fills = sum(step_fills)
        info = {
            "spread_pnl": pnls[0].spread,
            "inventory_pnl": pnls[0].inventory,
            "hedge_cost": pnls[0].hedge_cost,
            "total_pnl": pnls[0].total,
            "inventory": self.inventory[0],
            "competitor_pnl": pnls[1].total,
            "competitor_spread_pnl": pnls[1].spread,
            "competitor_inventory_pnl": pnls[1].inventory,
            "competitor_hedge_cost": pnls[1].hedge_cost,
            "competitor_inventory": self.inventory[1],
            "market_share": step_fills[0] / total_fills if total_fills else 0.0,
        }
        return self._observation(0), pnls[0].total, False, info
