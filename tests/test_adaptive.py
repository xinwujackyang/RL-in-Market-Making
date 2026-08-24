from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive import (
    DEFAULT_EPSILON_GRID,
    AdaptiveMarketMaker,
    AdaptiveResponseTable,
    MissingResponseStatistics,
    ResponseStatistics,
)


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


if __name__ == "__main__":
    unittest.main()
