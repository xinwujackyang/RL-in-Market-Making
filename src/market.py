from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import Config


@dataclass
class PnL:
    spread: float = 0.0
    inventory: float = 0.0
    hedge_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.inventory - self.hedge_cost


def evolve_mid_price(price: float, cfg: Config, rng: np.random.Generator) -> float:
    drift = (cfg.mu - 0.5 * cfg.sigma**2) * cfg.dt
    shock = cfg.sigma * math.sqrt(cfg.dt) * rng.standard_normal()
    return price * math.exp(drift + shock)


def generate_investor_orders(
    cfg: Config,
    rng: np.random.Generator,
    direction_rng: np.random.Generator | None = None,
) -> list[tuple[float, int]]:
    """Return (absolute size, dealer direction): +1 dealer buy, -1 dealer sell."""
    direction_rng = rng if direction_rng is None else direction_rng
    if cfg.order_size_mode == "gamma":
        sizes = rng.gamma(cfg.investor_shape, cfg.investor_scale, cfg.num_investors)
    else:
        sizes = np.ones(cfg.num_investors, dtype=float)
    # An investor buy is a dealer sell, hence the negative dealer direction.
    investor_buys = direction_rng.random(cfg.num_investors) < cfg.buy_probability
    directions = np.where(investor_buys, -1, 1)
    return list(zip(sizes.tolist(), directions.tolist()))


def reference_spread(price: float, size: float, cfg: Config) -> float:
    base = cfg.base_spread_bp * 1e-4 * price
    widen = cfg.lob_slope_bp * 1e-4 * price * math.sqrt(max(size, 1e-8))
    return base + widen


def investor_execution_price(
    price: float, size: float, dealer_direction: int, epsilon_bid: float, epsilon_ask: float, cfg: Config
) -> float:
    """Price paid by dealer: bid for a dealer buy, ask for a dealer sell."""
    spread = reference_spread(price, size, cfg)
    return price - spread * (1 + epsilon_bid) if dealer_direction > 0 else price + spread * (1 + epsilon_ask)


def execute_hedge(
    inventory: float, hedge_fraction: float, price: float, cfg: Config
) -> tuple[float, float]:
    hedge_size = -float(np.clip(hedge_fraction, 0.0, 1.0)) * inventory
    if abs(hedge_size) <= 1e-6:
        return inventory, 0.0
    spread = reference_spread(price, abs(hedge_size), cfg)
    hedge_price = price + spread if hedge_size > 0 else price - spread
    cost = hedge_size * (hedge_price - price)
    return inventory + hedge_size, cost
