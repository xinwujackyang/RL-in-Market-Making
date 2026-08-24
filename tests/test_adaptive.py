from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive import (
    DEFAULT_EPSILON_GRID,
    AdaptiveMarketMakerCompetitor,
    AdaptiveMarketMaker,
    AdaptiveResponseTable,
    MissingResponseStatistics,
    ResponseStatistics,
    cold_start_quotes,
)
from baselines import PersistentMarketMaker
from config import Config
from environments import SingleDealerMarketEnv, TwoDealerMarketEnv


def response(
    *,
    gross: float = 0.0,
    net: float = 0.0,
    net_variance: float = 0.0,
    spread_pnl: float = 0.0,
    spread_pnl_variance: float = 0.0,
) -> ResponseStatistics:
    return ResponseStatistics(
        mean_gross_volume=gross,
        mean_net_flow=net,
        variance_net_flow=net_variance,
        mean_normalized_spread_pnl=spread_pnl,
        variance_normalized_spread_pnl=spread_pnl_variance,
    )


def populated_statistics() -> dict[tuple[float, float], ResponseStatistics]:
    statistics = {
        (bid, ask): response()
        for bid in DEFAULT_EPSILON_GRID
        for ask in DEFAULT_EPSILON_GRID
    }
    # Symmetric quotes predict a share that decreases linearly from one to zero.
    for epsilon in DEFAULT_EPSILON_GRID:
        statistics[(epsilon, epsilon)] = response(gross=10.0 * (1.0 - epsilon))
    return statistics


class AdaptiveMarketMakerTests(unittest.TestCase):
    def test_step_one_selects_target_share_quote(self) -> None:
        table = AdaptiveResponseTable(populated_statistics())
        maker = AdaptiveMarketMaker(table, market_share_target=0.5, risk_aversion=1.0)

        self.assertAlmostEqual(maker.select_market_share_quote(market_volume=20.0), 0.0)

    def test_step_one_tolerance_selects_widest_eligible_quote(self) -> None:
        statistics = populated_statistics()
        statistics[(0.2, 0.2)] = response(gross=9.0)
        maker = AdaptiveMarketMaker(
            AdaptiveResponseTable(statistics), market_share_target=0.5, risk_aversion=1.0
        )

        self.assertAlmostEqual(maker.select_market_share_quote(market_volume=20.0), 0.2)

    def test_short_inventory_lowers_only_bid(self) -> None:
        statistics = populated_statistics()
        statistics[(-0.4, 0.0)] = response(net=5.0)
        maker = AdaptiveMarketMaker(
            AdaptiveResponseTable(statistics), market_share_target=0.5, risk_aversion=1.0
        )

        quote = maker.select_skewed_quote(
            base_epsilon=0.0,
            inventory=-5.0,
            reference_spread_at_zero=0.0,
            normal_volatility=1.0,
        )

        self.assertEqual(quote, (-0.4, 0.0))

    def test_long_inventory_lowers_only_ask(self) -> None:
        statistics = populated_statistics()
        statistics[(0.0, -0.6)] = response(net=-5.0)
        maker = AdaptiveMarketMaker(
            AdaptiveResponseTable(statistics), market_share_target=0.5, risk_aversion=1.0
        )

        quote = maker.select_skewed_quote(
            base_epsilon=0.0,
            inventory=5.0,
            reference_spread_at_zero=0.0,
            normal_volatility=1.0,
        )

        self.assertEqual(quote, (0.0, -0.6))

    def test_higher_gamma_gives_inventory_risk_more_weight(self) -> None:
        statistics = populated_statistics()
        statistics[(0.0, 0.0)] = response(net=0.0, spread_pnl=10.0)
        statistics[(0.0, -0.4)] = response(net=-5.0, spread_pnl=0.0)
        table = AdaptiveResponseTable(statistics)
        risk_neutral = AdaptiveMarketMaker(table, market_share_target=0.5, risk_aversion=0.0)
        risk_averse = AdaptiveMarketMaker(table, market_share_target=0.5, risk_aversion=3.0)

        neutral_quote = risk_neutral.select_skewed_quote(0.0, 5.0, 1.0, 1.0)
        averse_quote = risk_averse.select_skewed_quote(0.0, 5.0, 1.0, 1.0)

        self.assertEqual(neutral_quote, (0.0, 0.0))
        self.assertEqual(averse_quote, (0.0, -0.4))

    def test_hedge_uses_final_skewed_response(self) -> None:
        statistics = populated_statistics()
        # The final skewed quote predicts a buy flow that fully offsets the short.
        statistics[(-0.4, 0.0)] = response(net=5.0)
        # The symmetric quote would instead imply a full hedge with zero spread cost.
        statistics[(0.0, 0.0)] = response(gross=10.0, net=0.0)
        maker = AdaptiveMarketMaker(
            AdaptiveResponseTable(statistics), market_share_target=0.5, risk_aversion=1.0
        )

        decision = maker.decide(
            inventory=-5.0,
            market_volume=20.0,
            reference_spread_at_zero=0.0,
            normal_volatility=1.0,
            hedge_spread=lambda _: 0.0,
        )

        self.assertEqual((decision.epsilon_bid, decision.epsilon_ask), (-0.4, 0.0))
        self.assertEqual(decision.hedge_fraction, 0.0)

    def test_zero_inventory_does_not_skew(self) -> None:
        maker = AdaptiveMarketMaker(
            AdaptiveResponseTable(populated_statistics()),
            market_share_target=0.5,
            risk_aversion=1.0,
        )
        self.assertEqual(maker.select_skewed_quote(0.2, 0.0, 1.0, 1.0), (0.2, 0.2))

    def test_missing_cell_has_no_implicit_cold_start_fallback(self) -> None:
        table = AdaptiveResponseTable({(0.0, 0.0): response()})
        with self.assertRaisesRegex(MissingResponseStatistics, "cold-start behavior"):
            table.lookup(-0.2, 0.0)


class AdaptiveResponseTableUpdateTests(unittest.TestCase):
    def test_first_update_initializes_cell_directly(self) -> None:
        table = AdaptiveResponseTable()

        table.update(0.2, -0.4, 3.0, -2.0, 5.0)

        statistics = table.lookup(0.2, -0.4)
        self.assertEqual(statistics.mean_gross_volume, 3.0)
        self.assertEqual(statistics.mean_net_flow, -2.0)
        self.assertEqual(statistics.variance_net_flow, 0.0)
        self.assertEqual(statistics.mean_normalized_spread_pnl, 5.0)
        self.assertEqual(statistics.variance_normalized_spread_pnl, 0.0)

    def test_second_update_uses_ema_means_and_second_moments(self) -> None:
        table = AdaptiveResponseTable()
        table.update(0.2, -0.4, 10.0, 2.0, 4.0)

        table.update(0.2, -0.4, 4.0, -2.0, 2.0)

        statistics = table.lookup(0.2, -0.4)
        self.assertAlmostEqual(statistics.mean_gross_volume, 6.1)
        self.assertAlmostEqual(statistics.mean_net_flow, -0.6)
        self.assertAlmostEqual(statistics.variance_net_flow, 3.64)
        self.assertAlmostEqual(statistics.mean_normalized_spread_pnl, 2.7)
        self.assertAlmostEqual(statistics.variance_normalized_spread_pnl, 0.91)

    def test_update_changes_only_executed_cell(self) -> None:
        table = AdaptiveResponseTable()
        table.update(0.0, 0.0, 1.0, 2.0, 3.0)
        table.update(0.2, 0.0, 4.0, 5.0, 6.0)
        untouched_before = table.lookup(0.2, 0.0)

        table.update(0.0, 0.0, 7.0, 8.0, 9.0)

        self.assertEqual(table.lookup(0.2, 0.0), untouched_before)

    def test_missing_cell_still_fails_fast(self) -> None:
        table = AdaptiveResponseTable()
        table.update(0.0, 0.0, 1.0, 2.0, 3.0)

        with self.assertRaises(MissingResponseStatistics):
            table.lookup(0.2, 0.0)


class ColdStartQuoteTests(unittest.TestCase):
    def test_sequence_has_121_unique_cells(self) -> None:
        quotes = cold_start_quotes()
        self.assertEqual(len(quotes), 121)
        self.assertEqual(len(set(quotes)), 121)

    def test_first_eleven_quotes_are_diagonal(self) -> None:
        quotes = cold_start_quotes()
        expected = tuple((epsilon, epsilon) for epsilon in DEFAULT_EPSILON_GRID)
        self.assertEqual(quotes[:11], expected)

    def test_sequence_covers_complete_joint_grid(self) -> None:
        quotes = cold_start_quotes()
        expected = {
            (epsilon_bid, epsilon_ask)
            for epsilon_bid in DEFAULT_EPSILON_GRID
            for epsilon_ask in DEFAULT_EPSILON_GRID
        }
        self.assertEqual(set(quotes), expected)


class AdaptiveSimulatorIntegrationTests(unittest.TestCase):
    def test_single_dealer_environment_is_unchanged(self) -> None:
        environment = SingleDealerMarketEnv(Config(order_size_mode="unit"), seed=100)
        self.assertEqual(environment.t, 0)

    def test_cold_start_completes_then_adaptive_action_runs(self) -> None:
        adaptive = AdaptiveMarketMakerCompetitor(
            market_share_target=0.5,
            risk_aversion=1.0,
        )
        cfg = Config(order_size_mode="unit", sigma=0.0)
        env = TwoDealerMarketEnv(adaptive, cfg, seed=101)
        persistent_action = PersistentMarketMaker(0.5, 0.5, 0.0).act()

        for _ in range(121):
            _, _, _, info = env.step(persistent_action)
            self.assertTrue(info["competitor_cold_start"])

        self.assertTrue(adaptive.cold_start_complete)
        adaptive.response_table.validate_complete()

        _, _, _, info = env.step(persistent_action)
        action = np.array(
            [
                info["competitor_epsilon_bid"],
                info["competitor_epsilon_ask"],
                info["competitor_hedge_fraction"],
            ]
        )
        self.assertFalse(info["competitor_cold_start"])
        self.assertTrue(np.all(action[:2] >= -1.0))
        self.assertTrue(np.all(action[:2] <= 1.0))
        self.assertTrue(0.0 <= action[2] <= 1.0)


if __name__ == "__main__":
    unittest.main()
