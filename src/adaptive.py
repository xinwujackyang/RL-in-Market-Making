from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


DEFAULT_EPSILON_GRID = tuple(round(float(value), 1) for value in np.linspace(-1.0, 1.0, 11))
DEFAULT_HEDGE_GRID = tuple(round(float(value), 2) for value in np.linspace(0.0, 1.0, 101))
OLD_ESTIMATE_RETENTION = 0.35


@dataclass(frozen=True)
class ResponseStatistics:
    """Empirical response for one joint (epsilon_bid, epsilon_ask) quote."""

    mean_gross_volume: float
    mean_net_flow: float
    variance_net_flow: float
    mean_normalized_spread_pnl: float
    variance_normalized_spread_pnl: float

    def __post_init__(self) -> None:
        if self.variance_net_flow < 0.0:
            raise ValueError("variance_net_flow must be nonnegative")
        if self.variance_normalized_spread_pnl < 0.0:
            raise ValueError("variance_normalized_spread_pnl must be nonnegative")


class MissingResponseStatistics(ValueError):
    """Raised when decision logic encounters an unpopulated response-table cell."""


@dataclass
class _ResponseMoments:
    mean_gross_volume: float
    mean_net_flow: float
    second_moment_net_flow: float
    mean_normalized_spread_pnl: float
    second_moment_normalized_spread_pnl: float

    @classmethod
    def from_statistics(cls, statistics: ResponseStatistics) -> "_ResponseMoments":
        return cls(
            mean_gross_volume=statistics.mean_gross_volume,
            mean_net_flow=statistics.mean_net_flow,
            second_moment_net_flow=(
                statistics.variance_net_flow + statistics.mean_net_flow**2
            ),
            mean_normalized_spread_pnl=statistics.mean_normalized_spread_pnl,
            second_moment_normalized_spread_pnl=(
                statistics.variance_normalized_spread_pnl
                + statistics.mean_normalized_spread_pnl**2
            ),
        )

    def snapshot(self) -> ResponseStatistics:
        return ResponseStatistics(
            mean_gross_volume=self.mean_gross_volume,
            mean_net_flow=self.mean_net_flow,
            variance_net_flow=max(
                self.second_moment_net_flow - self.mean_net_flow**2,
                0.0,
            ),
            mean_normalized_spread_pnl=self.mean_normalized_spread_pnl,
            variance_normalized_spread_pnl=max(
                self.second_moment_normalized_spread_pnl
                - self.mean_normalized_spread_pnl**2,
                0.0,
            ),
        )


def cold_start_quotes(
    epsilon_grid: Sequence[float] = DEFAULT_EPSILON_GRID,
) -> tuple[tuple[float, float], ...]:
    """Return a deterministic full-grid pass, diagonal quotes first."""
    grid = tuple(float(value) for value in epsilon_grid)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("epsilon_grid must be nonempty and contain unique values")
    diagonal = tuple((epsilon, epsilon) for epsilon in grid)
    off_diagonal = tuple(
        (epsilon_bid, epsilon_ask)
        for epsilon_bid in grid
        for epsilon_ask in grid
        if epsilon_bid != epsilon_ask
    )
    return diagonal + off_diagonal


class AdaptiveResponseTable:
    """Online empirical moments for joint quote responses.

    Only an explicitly updated cell changes. Unvisited cells have no prior and
    fail fast at lookup time.
    """

    def __init__(
        self,
        statistics: Mapping[tuple[float, float], ResponseStatistics] | None = None,
        epsilon_grid: Sequence[float] = DEFAULT_EPSILON_GRID,
    ) -> None:
        grid = tuple(float(value) for value in epsilon_grid)
        if not grid or any(right <= left for left, right in zip(grid, grid[1:])):
            raise ValueError("epsilon_grid must be nonempty and strictly increasing")
        if grid[0] < -1.0 or grid[-1] > 1.0:
            raise ValueError("epsilon_grid must lie within [-1, 1]")

        self.epsilon_grid = grid
        self._moments = {
            (self._canonical_epsilon(bid), self._canonical_epsilon(ask)): (
                _ResponseMoments.from_statistics(value)
            )
            for (bid, ask), value in (statistics or {}).items()
        }

    def _canonical_epsilon(self, epsilon: float) -> float:
        matches = [
            value
            for value in self.epsilon_grid
            if math.isclose(epsilon, value, rel_tol=0.0, abs_tol=1e-6)
        ]
        if not matches:
            raise ValueError(f"epsilon {epsilon} is not on the configured grid")
        return matches[0]

    def lookup(self, epsilon_bid: float, epsilon_ask: float) -> ResponseStatistics:
        key = (self._canonical_epsilon(epsilon_bid), self._canonical_epsilon(epsilon_ask))
        try:
            return self._moments[key].snapshot()
        except KeyError as error:
            raise MissingResponseStatistics(
                f"response-table cell {key} is unpopulated; cold-start behavior is intentionally undefined"
            ) from error

    def update(
        self,
        epsilon_bid: float,
        epsilon_ask: float,
        gross_volume: float,
        net_flow: float,
        normalized_spread_pnl: float,
    ) -> None:
        """Update the executed quote cell using beta=0.35 old-estimate retention."""
        key = (self._canonical_epsilon(epsilon_bid), self._canonical_epsilon(epsilon_ask))
        gross_volume = float(gross_volume)
        net_flow = float(net_flow)
        normalized_spread_pnl = float(normalized_spread_pnl)

        if key not in self._moments:
            self._moments[key] = _ResponseMoments(
                mean_gross_volume=gross_volume,
                mean_net_flow=net_flow,
                second_moment_net_flow=net_flow**2,
                mean_normalized_spread_pnl=normalized_spread_pnl,
                second_moment_normalized_spread_pnl=normalized_spread_pnl**2,
            )
            return

        moments = self._moments[key]
        new_weight = 1.0 - OLD_ESTIMATE_RETENTION
        moments.mean_gross_volume = (
            OLD_ESTIMATE_RETENTION * moments.mean_gross_volume
            + new_weight * gross_volume
        )
        moments.mean_net_flow = (
            OLD_ESTIMATE_RETENTION * moments.mean_net_flow + new_weight * net_flow
        )
        moments.second_moment_net_flow = (
            OLD_ESTIMATE_RETENTION * moments.second_moment_net_flow
            + new_weight * net_flow**2
        )
        moments.mean_normalized_spread_pnl = (
            OLD_ESTIMATE_RETENTION * moments.mean_normalized_spread_pnl
            + new_weight * normalized_spread_pnl
        )
        moments.second_moment_normalized_spread_pnl = (
            OLD_ESTIMATE_RETENTION * moments.second_moment_normalized_spread_pnl
            + new_weight * normalized_spread_pnl**2
        )

    def validate_complete(self) -> None:
        missing = [
            (bid, ask)
            for bid in self.epsilon_grid
            for ask in self.epsilon_grid
            if (bid, ask) not in self._moments
        ]
        if missing:
            raise MissingResponseStatistics(
                f"response table is missing {len(missing)} of {len(self.epsilon_grid) ** 2} cells"
            )


@dataclass(frozen=True)
class AdaptiveDecision:
    base_epsilon: float
    epsilon_bid: float
    epsilon_ask: float
    hedge_fraction: float

    @property
    def action(self) -> np.ndarray:
        return np.array(
            [self.epsilon_bid, self.epsilon_ask, self.hedge_fraction],
            dtype=np.float32,
        )


class AdaptiveMarketMakerCompetitor:
    """Minimal simulator adapter around the pure Adaptive MM decision logic."""

    def __init__(
        self,
        market_share_target: float,
        risk_aversion: float,
    ) -> None:
        self.market_share_target = float(market_share_target)
        self.risk_aversion = float(risk_aversion)
        self.reset_adaptive_state()

    def reset_adaptive_state(self) -> None:
        self.response_table = AdaptiveResponseTable()
        self.market_maker = AdaptiveMarketMaker(
            self.response_table,
            market_share_target=self.market_share_target,
            risk_aversion=self.risk_aversion,
        )
        self._cold_start_sequence = cold_start_quotes()
        self._cold_start_index = 0
        self.last_base_epsilon = float("nan")
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_action_was_cold_start = True

    @property
    def cold_start_complete(self) -> bool:
        return self._cold_start_index >= len(self._cold_start_sequence)

    def act_with_market_state(
        self,
        *,
        inventory: float,
        price: float,
        previous_market_volume: float,
        reference_spread_at_zero: float,
        normal_volatility: float,
        hedge_spread: Callable[[float], float],
    ) -> np.ndarray:
        del price  # Included in the hook state for diagnostics and future extensions.
        if not self.cold_start_complete:
            epsilon_bid, epsilon_ask = self._cold_start_sequence[self._cold_start_index]
            self._cold_start_index += 1
            self.last_base_epsilon = float("nan")
            self.last_action_was_cold_start = True
            self.last_action = np.array(
                [epsilon_bid, epsilon_ask, 0.0], dtype=np.float32
            )
            return self.last_action.copy()

        self.response_table.validate_complete()
        decision = self.market_maker.decide(
            inventory=inventory,
            market_volume=previous_market_volume,
            reference_spread_at_zero=reference_spread_at_zero,
            normal_volatility=normal_volatility,
            hedge_spread=hedge_spread,
        )
        self.last_base_epsilon = decision.base_epsilon
        self.last_action_was_cold_start = False
        self.last_action = decision.action
        return self.last_action.copy()

    def observe_outcome(
        self,
        *,
        epsilon_bid: float,
        epsilon_ask: float,
        gross_volume: float,
        net_flow: float,
        spread_pnl: float,
        reference_spread_at_zero: float,
    ) -> None:
        if reference_spread_at_zero <= 0.0:
            raise ValueError("reference_spread_at_zero must be positive")
        self.response_table.update(
            epsilon_bid,
            epsilon_ask,
            gross_volume,
            net_flow,
            spread_pnl / reference_spread_at_zero,
        )


class AdaptiveMarketMaker:
    """Pure paper-inspired Adaptive MM decision logic over populated statistics."""

    def __init__(
        self,
        response_table: AdaptiveResponseTable,
        market_share_target: float,
        risk_aversion: float,
        market_share_tolerance: float = 0.05,
        hedge_grid: Sequence[float] = DEFAULT_HEDGE_GRID,
    ) -> None:
        if not 0.0 <= market_share_target <= 1.0:
            raise ValueError("market_share_target must lie in [0, 1]")
        if risk_aversion < 0.0:
            raise ValueError("risk_aversion must be nonnegative")
        if market_share_tolerance < 0.0:
            raise ValueError("market_share_tolerance must be nonnegative")

        hedge_candidates = tuple(float(value) for value in hedge_grid)
        if not hedge_candidates or any(
            right <= left for left, right in zip(hedge_candidates, hedge_candidates[1:])
        ):
            raise ValueError("hedge_grid must be nonempty and strictly increasing")
        if hedge_candidates[0] < 0.0 or hedge_candidates[-1] > 1.0:
            raise ValueError("hedge_grid must lie within [0, 1]")

        self.response_table = response_table
        self.market_share_target = float(market_share_target)
        self.risk_aversion = float(risk_aversion)
        self.market_share_tolerance = float(market_share_tolerance)
        self.hedge_grid = hedge_candidates

    def select_market_share_quote(self, market_volume: float) -> float:
        """Step 1: choose the widest near-optimal symmetric grid quote."""
        if market_volume <= 0.0:
            raise ValueError("market_volume must be positive")

        costs = {}
        for epsilon in self.response_table.epsilon_grid:
            response = self.response_table.lookup(epsilon, epsilon)
            predicted_share = response.mean_gross_volume / market_volume
            costs[epsilon] = abs(self.market_share_target - predicted_share)

        minimum_cost = min(costs.values())
        eligible = [
            epsilon
            for epsilon, cost in costs.items()
            if abs(cost - minimum_cost) <= self.market_share_tolerance + 1e-12
        ]
        return max(eligible)

    def select_skewed_quote(
        self,
        base_epsilon: float,
        inventory: float,
        reference_spread_at_zero: float,
        normal_volatility: float,
    ) -> tuple[float, float]:
        """Step 2: decrease only the inventory-correcting quote side."""
        if reference_spread_at_zero < 0.0:
            raise ValueError("reference_spread_at_zero must be nonnegative")
        if normal_volatility < 0.0:
            raise ValueError("normal_volatility must be nonnegative")
        if inventory == 0.0:
            return base_epsilon, base_epsilon

        candidates = [
            epsilon
            for epsilon in self.response_table.epsilon_grid
            if epsilon <= base_epsilon + 1e-12
        ]
        quote_costs: list[tuple[float, float]] = []
        for epsilon in candidates:
            pair = (epsilon, base_epsilon) if inventory < 0.0 else (base_epsilon, epsilon)
            response = self.response_table.lookup(*pair)
            expected_inventory_square = (
                inventory + response.mean_net_flow
            ) ** 2 + response.variance_net_flow
            risk_variance = (
                reference_spread_at_zero**2 * response.variance_normalized_spread_pnl
                + normal_volatility**2 * expected_inventory_square
            )
            cost = (
                -reference_spread_at_zero * response.mean_normalized_spread_pnl
                + self.risk_aversion * math.sqrt(max(risk_variance, 0.0))
            )
            quote_costs.append((cost, epsilon))

        minimum_cost = min(cost for cost, _ in quote_costs)
        # Prefer the least aggressive quote when objective values tie.
        skew_epsilon = max(
            epsilon for cost, epsilon in quote_costs if math.isclose(cost, minimum_cost, abs_tol=1e-12)
        )
        return (
            (skew_epsilon, base_epsilon)
            if inventory < 0.0
            else (base_epsilon, skew_epsilon)
        )

    def select_hedge_fraction(
        self,
        inventory: float,
        final_quote: tuple[float, float],
        normal_volatility: float,
        hedge_spread: Callable[[float], float],
    ) -> float:
        """Choose x using flow predicted for the final, post-skew quote pair.

        ``hedge_spread`` receives the signed hedge trade ``-x * inventory`` so
        callers can select the appropriate side of an asymmetric reference book.
        """
        if normal_volatility < 0.0:
            raise ValueError("normal_volatility must be nonnegative")
        response = self.response_table.lookup(*final_quote)

        hedge_costs: list[tuple[float, float]] = []
        for fraction in self.hedge_grid:
            hedge_trade = -fraction * inventory
            spread = float(hedge_spread(hedge_trade))
            if spread < 0.0:
                raise ValueError("hedge_spread must return a nonnegative spread")
            expected_inventory_square = (
                inventory * (1.0 - fraction) + response.mean_net_flow
            ) ** 2 + response.variance_net_flow
            cost = (
                abs(hedge_trade) * spread
                + self.risk_aversion
                * normal_volatility
                * math.sqrt(max(expected_inventory_square, 0.0))
            )
            hedge_costs.append((cost, fraction))

        minimum_cost = min(cost for cost, _ in hedge_costs)
        # Avoid unnecessary hedge turnover when objective values tie.
        return min(
            fraction
            for cost, fraction in hedge_costs
            if math.isclose(cost, minimum_cost, abs_tol=1e-12)
        )

    def decide(
        self,
        inventory: float,
        market_volume: float,
        reference_spread_at_zero: float,
        normal_volatility: float,
        hedge_spread: Callable[[float], float],
    ) -> AdaptiveDecision:
        base_epsilon = self.select_market_share_quote(market_volume)
        final_quote = self.select_skewed_quote(
            base_epsilon,
            inventory,
            reference_spread_at_zero,
            normal_volatility,
        )
        hedge_fraction = self.select_hedge_fraction(
            inventory,
            final_quote,
            normal_volatility,
            hedge_spread,
        )
        return AdaptiveDecision(
            base_epsilon=base_epsilon,
            epsilon_bid=final_quote[0],
            epsilon_ask=final_quote[1],
            hedge_fraction=hedge_fraction,
        )
