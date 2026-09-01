from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import Config
from market import reference_spread


@dataclass(frozen=True)
class AvellanedaStoikovQuote:
    """One A-S decision, retaining both analytical and executable quotes."""

    inventory: float
    price: float
    price_variance: float
    reference_distance: float
    raw_epsilon_bid: float
    raw_epsilon_ask: float
    epsilon_bid: float
    epsilon_ask: float
    bid_clipped: bool
    ask_clipped: bool


class AvellanedaStoikovMarketMaker:
    """Stationary normalized A-S benchmark with native simulator execution.

    The canonical A-S reservation price and total spread are

        r = S - q * gamma * V
        Delta = gamma * V + 2 / gamma * log(1 + gamma / k),

    where ``V = S**2 * sigma**2 * dt * risk_horizon_steps`` is the local
    Brownian-price variance over a fixed receding risk horizon.  This implies

        delta_bid = q * gamma * V + Delta / 2
        delta_ask = -q * gamma * V + Delta / 2.

    By default the controller uses the scale-normalized benchmark described in
    Phase 13.  It fixes ``rho = gamma * D`` and ``kappa = k * D``, where ``D``
    is the simulator's unit-flow reference quote distance.  The two anchors are
    a neutral quote ``epsilon=neutral_epsilon`` and a one-reference-distance
    inventory shift at ``inventory_anchor``.  Passing both ``gamma`` and ``k``
    instead selects the canonical fixed-dollar-parameter form.
    """

    def __init__(
        self,
        gamma: float | None = None,
        k: float | None = None,
        risk_horizon_steps: int = 26,
        *,
        inventory_anchor: float = 20.0,
        neutral_epsilon: float = 0.0,
        hedge_fraction: float = 0.0,
        cfg: Config | None = None,
        clip_actions: bool = True,
    ) -> None:
        if (gamma is None) != (k is None):
            raise ValueError("gamma and k must either both be supplied or both omitted")
        if gamma is not None and gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if k is not None and k <= 0.0:
            raise ValueError("k must be positive")
        if risk_horizon_steps <= 0:
            raise ValueError("risk_horizon_steps must be positive")
        if inventory_anchor <= 0.0:
            raise ValueError("inventory_anchor must be positive")
        if neutral_epsilon <= -1.0:
            raise ValueError("neutral_epsilon must exceed -1")
        if not 0.0 <= hedge_fraction <= 1.0:
            raise ValueError("hedge_fraction must lie in [0, 1]")

        self.cfg = cfg or Config()
        self.gamma = None if gamma is None else float(gamma)
        self.k = None if k is None else float(k)
        self.risk_horizon_steps = int(risk_horizon_steps)
        self.inventory_anchor = float(inventory_anchor)
        self.neutral_epsilon = float(neutral_epsilon)
        self.hedge_fraction = float(hedge_fraction)
        self.clip_actions = bool(clip_actions)

        self.parameterization = "normalized" if gamma is None else "canonical"
        self.rho: float | None = None
        self.kappa: float | None = None
        if self.parameterization == "normalized":
            # V / D^2 is price-invariant under GBM and proportional spreads.
            price = self.cfg.P0
            distance = self.reference_distance(price)
            nu = self.price_variance(price) / distance**2
            inventory_shift = 1.0 / self.inventory_anchor
            self.rho = inventory_shift / nu
            exponent = self.rho * (
                1.0 + self.neutral_epsilon - inventory_shift / 2.0
            )
            denominator = np.expm1(exponent)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("anchors imply an invalid normalized kappa")
            self.kappa = float(self.rho / denominator)

        self.reset_adaptive_state()

    def reset_adaptive_state(self) -> None:
        """Reset diagnostics without changing the frozen policy parameters."""

        self.action_count = 0
        self.bid_clip_count = 0
        self.ask_clip_count = 0
        self.last_quote: AvellanedaStoikovQuote | None = None
        # The environment's market-state hook exposes these diagnostic fields
        # for its original Adaptive MM user.  They remain false/non-probing for
        # A-S; ``last_base_epsilon`` is the current symmetric raw quote level.
        self.last_base_epsilon = self.neutral_epsilon
        self.last_action_was_cold_start = False
        self.last_action_was_probe = False

    def price_variance(self, price: float) -> float:
        """Local GBM/Brownian variance over the fixed receding risk horizon."""

        if price <= 0.0:
            raise ValueError("price must be positive")
        return (
            float(price) ** 2
            * self.cfg.sigma**2
            * self.cfg.dt
            * self.risk_horizon_steps
        )

    def reference_distance(self, price: float) -> float:
        """Simulator reference quote distance for a unit investor order."""

        return reference_spread(float(price), 1.0, self.cfg)

    def normalized_coordinates(self, price: float) -> tuple[float, float, float]:
        """Return ``(nu, rho, kappa)`` at the current price scale."""

        distance = self.reference_distance(price)
        nu = self.price_variance(price) / distance**2
        if self.parameterization == "normalized":
            assert self.rho is not None and self.kappa is not None
            return nu, self.rho, self.kappa
        assert self.gamma is not None and self.k is not None
        return nu, self.gamma * distance, self.k * distance

    def canonical_parameters(self, price: float) -> tuple[float, float]:
        """Return the dollar-coordinate ``(gamma, k)`` at ``price``."""

        if self.parameterization == "canonical":
            assert self.gamma is not None and self.k is not None
            return self.gamma, self.k
        distance = self.reference_distance(price)
        assert self.rho is not None and self.kappa is not None
        return self.rho / distance, self.kappa / distance

    def quote(self, inventory: float, price: float) -> AvellanedaStoikovQuote:
        """Compute one quote and retain whether either action bound was active."""

        inventory = float(inventory)
        price = float(price)
        variance = self.price_variance(price)
        distance = self.reference_distance(price)

        if self.parameterization == "normalized":
            nu, rho, kappa = self.normalized_coordinates(price)
            symmetric_distance = rho * nu / 2.0 + math.log1p(rho / kappa) / rho
            inventory_shift = inventory * rho * nu
            raw_bid = symmetric_distance + inventory_shift - 1.0
            raw_ask = symmetric_distance - inventory_shift - 1.0
        else:
            assert self.gamma is not None and self.k is not None
            total_spread = self.gamma * variance + (
                2.0 / self.gamma * math.log1p(self.gamma / self.k)
            )
            delta_bid = inventory * self.gamma * variance + total_spread / 2.0
            delta_ask = -inventory * self.gamma * variance + total_spread / 2.0
            raw_bid = delta_bid / distance - 1.0
            raw_ask = delta_ask / distance - 1.0

        bid_clipped = not -1.0 <= raw_bid <= 1.0
        ask_clipped = not -1.0 <= raw_ask <= 1.0
        if self.clip_actions:
            epsilon_bid = float(np.clip(raw_bid, -1.0, 1.0))
            epsilon_ask = float(np.clip(raw_ask, -1.0, 1.0))
        else:
            epsilon_bid = float(raw_bid)
            epsilon_ask = float(raw_ask)

        decision = AvellanedaStoikovQuote(
            inventory=inventory,
            price=price,
            price_variance=variance,
            reference_distance=distance,
            raw_epsilon_bid=float(raw_bid),
            raw_epsilon_ask=float(raw_ask),
            epsilon_bid=epsilon_bid,
            epsilon_ask=epsilon_ask,
            bid_clipped=bid_clipped,
            ask_clipped=ask_clipped,
        )
        self.action_count += 1
        self.bid_clip_count += int(bid_clipped)
        self.ask_clip_count += int(ask_clipped)
        self.last_quote = decision
        self.last_base_epsilon = float((raw_bid + raw_ask) / 2.0)
        return decision

    def act_with_market_state(
        self,
        *,
        inventory: float,
        price: float,
        **market_state: object,
    ) -> np.ndarray:
        """Environment hook: A-S quotes, simulator-native routing, no hedge."""

        del market_state
        quote = self.quote(inventory, price)
        return np.array(
            [quote.epsilon_bid, quote.epsilon_ask, self.hedge_fraction],
            dtype=np.float32,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Fallback for the default observation layout ``[q, relative_price, ...]``."""

        if len(observation) < 2:
            raise ValueError("observation must contain inventory and price")
        inventory = float(observation[0])
        price_feature = float(observation[1])
        price = (
            self.cfg.P0 * (1.0 + price_feature)
            if self.cfg.relative_price
            else price_feature
        )
        quote = self.quote(inventory, price)
        return np.array(
            [quote.epsilon_bid, quote.epsilon_ask, self.hedge_fraction],
            dtype=np.float32,
        )

    @property
    def bid_clip_frequency(self) -> float:
        return self.bid_clip_count / self.action_count if self.action_count else 0.0

    @property
    def ask_clip_frequency(self) -> float:
        return self.ask_clip_count / self.action_count if self.action_count else 0.0

    @property
    def side_clip_frequency(self) -> float:
        if not self.action_count:
            return 0.0
        return (self.bid_clip_count + self.ask_clip_count) / (2 * self.action_count)
