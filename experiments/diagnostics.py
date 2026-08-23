from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Config
from market import PnL, execute_hedge, generate_investor_orders, investor_execution_price, reference_spread
from networks import ActorCritic, SquashedNormal
from ppo import RolloutBuffer, discount_cumsum


def main() -> None:
    cfg = Config()
    price, next_price, size = 100.0, 101.0, 1.0
    spread = reference_spread(price, size, cfg)

    ask = investor_execution_price(price, size, -1, 0.0, 0.0, cfg)
    sell_inventory = -size
    sell_inv_pnl = (next_price - price) * sell_inventory
    assert ask == price + spread and sell_inventory < 0 and sell_inv_pnl < 0

    bid = investor_execution_price(price, size, +1, 0.0, 0.0, cfg)
    buy_inventory = size
    buy_inv_pnl = (next_price - price) * buy_inventory
    assert bid == price - spread and buy_inventory > 0 and buy_inv_pnl > 0

    positive_after, positive_cost = execute_hedge(10.0, 0.5, price, cfg)
    negative_after, negative_cost = execute_hedge(-10.0, 0.5, price, cfg)
    assert positive_after == 5.0 and negative_after == -5.0
    assert positive_cost > 0 and negative_cost > 0

    pnl = PnL(spread=2.0, inventory=-0.5, hedge_cost=0.25)
    assert pnl.total == 1.25

    unit_cfg = Config(num_investors=20, order_size_mode="unit")
    unit_orders = generate_investor_orders(unit_cfg, np.random.default_rng(7))
    assert len(unit_orders) == 20
    assert all(size == 1.0 and direction in {-1, 1} for size, direction in unit_orders)
    gamma_orders = generate_investor_orders(cfg, np.random.default_rng(7))
    assert len(gamma_orders) == 20 and any(size != 1.0 for size, _ in gamma_orders)

    mean = torch.zeros(64, 3)
    distribution = SquashedNormal(mean, torch.ones_like(mean))
    actions = distribution.sample()
    assert torch.all(actions[:, :2] >= -1) and torch.all(actions[:, :2] <= 1)
    assert torch.all(actions[:, 2] >= 0) and torch.all(actions[:, 2] <= 1)
    assert torch.isfinite(distribution.log_prob(actions)).all()
    same_ratio = (distribution.log_prob(actions).sum(-1) - distribution.log_prob(actions).sum(-1)).exp()
    torch.testing.assert_close(same_ratio, torch.ones_like(same_ratio))

    rewards = np.array([1.0, 2.0], dtype=np.float32)
    bootstrapped = discount_cumsum(np.append(rewards, 3.0), 0.9)[:-1]
    np.testing.assert_allclose(bootstrapped, [1 + 0.9 * 2 + 0.9**2 * 3, 2 + 0.9 * 3])

    gae_cfg = Config(horizon=2, gamma=0.9, gae_lambda=0.8)
    buffer = RolloutBuffer(1, 1, 2, gae_cfg)
    buffer.rewards[:] = [1.0, 2.0]
    buffer.values[:] = [0.5, 0.25]
    buffer.pointer = 2
    buffer.finish_path(last_value=0.75)
    deltas = np.array([1.0 + 0.9 * 0.25 - 0.5, 2.0 + 0.9 * 0.75 - 0.25])
    expected_advantages = np.array([deltas[0] + 0.9 * 0.8 * deltas[1], deltas[1]])
    np.testing.assert_allclose(buffer.advantages, expected_advantages, rtol=1e-6)

    network = ActorCritic(observation_dim=5, action_dim=3, hidden_size=16, hidden_layers=2)
    observations = torch.ones(8, 5)
    _, values = network.distribution_and_value(observations)
    values.square().mean().backward()
    actor_parameters = list(network.actor.parameters()) + list(network.policy_mean.parameters())
    critic_parameters = list(network.critic.parameters()) + list(network.value_head.parameters())
    assert all(parameter.grad is None for parameter in actor_parameters)
    assert any(parameter.grad is not None for parameter in critic_parameters)

    print(f"A investor buys: ask={ask:.6f}, dealer inventory={sell_inventory:.1f}, up-move InvPnL={sell_inv_pnl:.1f}")
    print(f"B investor sells: bid={bid:.6f}, dealer inventory={buy_inventory:.1f}, up-move InvPnL={buy_inv_pnl:.1f}")
    print(f"C positive hedge: 10 -> {positive_after:.1f}, cost={positive_cost:.6f}")
    print(f"D negative hedge: -10 -> {negative_after:.1f}, cost={negative_cost:.6f}")
    print("Gamma/unit flow, bounded-action, reward identity, bootstrap, and actor/critic gradient isolation diagnostics passed.")


if __name__ == "__main__":
    main()
