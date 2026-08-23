from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import PersistentMarketMaker
from config import Config
from environments import TwoDealerMarketEnv
from market import generate_investor_orders


class InvestorFlowTests(unittest.TestCase):
    def test_config_rejects_invalid_population_settings(self) -> None:
        for changes in (
            {"num_investors": 0},
            {"order_size_mode": "lognormal"},
            {"buy_probability": -0.1},
            {"buy_probability": 1.1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**changes)

    def test_unit_flow_has_twenty_independent_unit_orders(self) -> None:
        cfg = Config(num_investors=20, order_size_mode="unit", buy_probability=0.5)
        orders = generate_investor_orders(cfg, np.random.default_rng(1))
        self.assertEqual(len(orders), 20)
        self.assertTrue(all(size == 1.0 for size, _ in orders))
        self.assertTrue(all(direction in {-1, 1} for _, direction in orders))

    def test_buy_probability_uses_dealer_side_sign(self) -> None:
        buy_cfg = Config(num_investors=20, order_size_mode="unit", buy_probability=1.0)
        sell_cfg = Config(num_investors=20, order_size_mode="unit", buy_probability=0.0)
        self.assertTrue(
            all(direction == -1 for _, direction in generate_investor_orders(buy_cfg, np.random.default_rng(2)))
        )
        self.assertTrue(
            all(direction == 1 for _, direction in generate_investor_orders(sell_cfg, np.random.default_rng(2)))
        )

    def test_gamma_mode_is_preserved(self) -> None:
        cfg = Config(num_investors=20, order_size_mode="gamma")
        orders = generate_investor_orders(cfg, np.random.default_rng(3))
        self.assertEqual(len(orders), 20)
        self.assertTrue(any(size != 1.0 for size, _ in orders))
        self.assertTrue(all(size > 0.0 for size, _ in orders))

    def test_step_diagnostics_and_routing(self) -> None:
        cfg = Config(num_investors=20, order_size_mode="unit", sigma=0.0)
        competitor = PersistentMarketMaker(0.5, 0.5, 0.0)
        env = TwoDealerMarketEnv(competitor, cfg, seed=17)
        _, reward, _, info = env.step(np.array([-0.5, -0.5, 0.0], dtype=np.float32))
        self.assertEqual(info["n_buy"] + info["n_sell"], 20)
        self.assertEqual(info["n_rl_won"], 20)
        self.assertEqual(info["n_competitor_won"], 0)
        self.assertEqual(info["gross_investor_volume"], 20.0)
        self.assertEqual(info["net_investor_flow"], info["n_buy"] - info["n_sell"])
        self.assertEqual(info["market_share"], 1.0)
        self.assertAlmostEqual(
            reward,
            info["spread_pnl"] + info["inventory_pnl"] - info["hedge_cost"],
        )

    def test_flow_modes_share_directions_and_midprice_path(self) -> None:
        gamma_cfg = Config(order_size_mode="gamma")
        unit_cfg = gamma_cfg.updated(order_size_mode="unit")
        gamma_env = TwoDealerMarketEnv(PersistentMarketMaker(), gamma_cfg, seed=99)
        unit_env = TwoDealerMarketEnv(PersistentMarketMaker(), unit_cfg, seed=99)
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            gamma_observation, _, _, gamma_info = gamma_env.step(action)
            unit_observation, _, _, unit_info = unit_env.step(action)
            self.assertEqual(gamma_info["n_buy"], unit_info["n_buy"])
            self.assertEqual(gamma_info["n_sell"], unit_info["n_sell"])
            self.assertEqual(gamma_observation[1], unit_observation[1])

    def test_relative_price_is_zero_at_inception(self) -> None:
        raw_env = TwoDealerMarketEnv(PersistentMarketMaker(), Config(), seed=7)
        relative_env = TwoDealerMarketEnv(
            PersistentMarketMaker(), Config(relative_price=True), seed=7
        )
        self.assertEqual(raw_env.reset()[1], 100.0)
        self.assertEqual(relative_env.reset()[1], 0.0)

    def test_fill_feedback_uses_previous_step_side_specific_fills(self) -> None:
        cfg = Config(
            num_investors=20,
            order_size_mode="unit",
            sigma=0.0,
            include_fill_feedback=True,
        )
        env = TwoDealerMarketEnv(PersistentMarketMaker(), cfg, seed=23)
        initial_observation = env.reset()
        self.assertEqual(initial_observation.shape, (7,))
        self.assertEqual(initial_observation[5:].tolist(), [0.0, 0.0])

        next_observation, _, _, info = env.step(
            np.array([-0.5, -0.5, 0.0], dtype=np.float32)
        )
        self.assertEqual(info["n_rl_won"], 20)
        self.assertAlmostEqual(next_observation[5], info["n_sell"] / 20)
        self.assertAlmostEqual(next_observation[6], info["n_buy"] / 20)


if __name__ == "__main__":
    unittest.main()
