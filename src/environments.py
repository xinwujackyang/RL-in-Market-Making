from __future__ import annotations

import numpy as np

from config import Config
from market import PnL, evolve_mid_price, execute_hedge, generate_investor_orders, investor_execution_price, reference_spread


def _clip_action(action: np.ndarray) -> np.ndarray:
    return np.array(
        [np.clip(action[0], -1, 1), np.clip(action[1], -1, 1), np.clip(action[2], 0, 1)],
        dtype=np.float32,
    )


def _rng_streams(seed: int) -> tuple[np.random.Generator, ...]:
    """Independent streams keep paired flow experiments on common random paths."""
    return tuple(np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4))


class SingleDealerMarketEnv:
    """Single dealer environment extracted from MMRL_1MM.ipynb."""

    def __init__(self, cfg: Config | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or Config()
        base_seed = self.cfg.seed if seed is None else seed
        self.size_rng, self.direction_rng, self.price_rng, self.routing_rng = _rng_streams(base_seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.price = self.cfg.P0
        self.inventory = 0.0
        self.last_pnl = PnL()
        self.t = 0
        return self._observation()

    def _observation(self) -> np.ndarray:
        price_feature = self.price / self.cfg.P0 - 1.0 if self.cfg.relative_price else self.price
        return np.array(
            [self.inventory, price_feature, self.last_pnl.spread, self.last_pnl.inventory, self.last_pnl.hedge_cost],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        eps_bid, eps_ask, hedge_fraction = _clip_action(action)
        # The hedge acts on inventory known at decision time. Investor flow then
        # changes inventory, and that post-hedge/post-flow inventory is exposed
        # to P_t -> P_{t+1}, matching z_t(1-x_t) + v_t in Ganesh et al.
        self.inventory, hedge_cost = execute_hedge(self.inventory, hedge_fraction, self.price, self.cfg)
        spread_pnl = 0.0
        for size, direction in generate_investor_orders(self.cfg, self.size_rng, self.direction_rng):
            epsilon = eps_bid if direction == 1 else eps_ask
            spread_pnl += size * reference_spread(self.price, size, self.cfg) * (1 + epsilon)
            self.inventory += direction * size

        old_price = self.price
        self.price = evolve_mid_price(self.price, self.cfg, self.price_rng)
        inventory_pnl = (self.price - old_price) * self.inventory
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

    def __init__(self, competitor, cfg: Config | None = None, seed: int | None = None, reward_mode: str = "total") -> None:
        self.cfg = cfg or Config()
        self.competitor = competitor
        if reward_mode not in {"total", "spread"}:
            raise ValueError("reward_mode must be 'total' or 'spread'")
        self.reward_mode = reward_mode
        base_seed = self.cfg.seed if seed is None else seed
        self.size_rng, self.direction_rng, self.price_rng, self.routing_rng = _rng_streams(base_seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.price = self.cfg.P0
        self.inventory = [0.0, 0.0]
        self.last_pnl = [PnL(), PnL()]
        self.market_share = [0.0, 0.0]
        self.t = 0
        return self._observation(0)

    def _observation(self, dealer: int) -> np.ndarray:
        pnl = self.last_pnl[dealer]
        price_feature = self.price / self.cfg.P0 - 1.0 if self.cfg.relative_price else self.price
        return np.array(
            [self.inventory[dealer], price_feature, pnl.total, pnl.inventory, pnl.hedge_cost],
            dtype=np.float32,
        )

    def step(self, rl_action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        actions = [_clip_action(rl_action), _clip_action(self.competitor.act(self._observation(1)))]
        spread_pnl = [0.0, 0.0]
        step_volume = [0.0, 0.0]
        step_orders = [0, 0]
        hedge_costs = [0.0, 0.0]
        investor_buy_count = investor_sell_count = 0
        investor_buy_volume = investor_sell_volume = 0.0

        # Hedge only the inventory visible when each action is selected.
        for dealer, action in enumerate(actions):
            self.inventory[dealer], hedge_costs[dealer] = execute_hedge(
                self.inventory[dealer], action[2], self.price, self.cfg
            )

        for size, direction in generate_investor_orders(self.cfg, self.size_rng, self.direction_rng):
            if direction < 0:
                investor_buy_count += 1
                investor_buy_volume += size
            else:
                investor_sell_count += 1
                investor_sell_volume += size
            ref = reference_spread(self.price, size, self.cfg)
            prices = [
                investor_execution_price(self.price, size, direction, action[0], action[1], self.cfg)
                for action in actions
            ]
            if prices[0] == prices[1]:
                winner = int(self.routing_rng.random() >= 0.5)
            elif direction > 0:  # investor sells; highest dealer bid wins
                winner = int(prices[1] > prices[0])
            else:
                winner = int(prices[1] < prices[0])  # investor buys; lowest ask wins
            epsilon = actions[winner][0 if direction > 0 else 1]
            spread_pnl[winner] += size * ref * (1 + epsilon)
            self.inventory[winner] += direction * size
            step_volume[winner] += size
            step_orders[winner] += 1

        old_price = self.price
        self.price = evolve_mid_price(self.price, self.cfg, self.price_rng)
        pnls = []
        for dealer in range(2):
            inv_pnl = (self.price - old_price) * self.inventory[dealer]
            pnls.append(PnL(spread_pnl[dealer], inv_pnl, hedge_costs[dealer]))
            self.market_share[dealer] += step_volume[dealer]
        self.last_pnl = pnls
        self.t += 1
        total_volume = sum(step_volume)
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
            "market_share": step_volume[0] / total_volume if total_volume else 0.0,
            "n_buy": investor_buy_count,
            "n_sell": investor_sell_count,
            "n_rl_won": step_orders[0],
            "n_competitor_won": step_orders[1],
            "gross_investor_volume": investor_buy_volume + investor_sell_volume,
            "net_investor_flow": investor_buy_volume - investor_sell_volume,
        }
        reward = pnls[0].total if self.reward_mode == "total" else pnls[0].spread
        return self._observation(0), reward, False, info
