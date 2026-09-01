from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avellaneda_stoikov import AvellanedaStoikovMarketMaker
from config import Config


def main() -> None:
    cfg = Config(order_size_mode="unit", relative_price=True, sigma=0.2)
    maker = AvellanedaStoikovMarketMaker(cfg=cfg)

    neutral = maker.quote(0.0, cfg.P0)
    assert np.isclose(neutral.raw_epsilon_bid, 0.0, atol=1e-12)
    assert np.isclose(neutral.raw_epsilon_ask, 0.0, atol=1e-12)

    for inventory in (-5.0, -1.0, 1.0, 5.0):
        quote = maker.quote(inventory, cfg.P0)
        expected_skew = 0.1 * inventory
        assert np.isclose(
            quote.raw_epsilon_bid - quote.raw_epsilon_ask,
            expected_skew,
            atol=1e-12,
        )
        assert np.isclose(
            (quote.raw_epsilon_bid + quote.raw_epsilon_ask) / 2.0,
            0.0,
            atol=1e-12,
        )
        if inventory > 0.0:
            assert quote.raw_epsilon_bid > quote.raw_epsilon_ask
        else:
            assert quote.raw_epsilon_bid < quote.raw_epsilon_ask

    low_price = maker.quote(3.0, 80.0)
    high_price = maker.quote(3.0, 160.0)
    assert np.isclose(
        low_price.raw_epsilon_bid, high_price.raw_epsilon_bid, atol=1e-12
    )
    assert np.isclose(
        low_price.raw_epsilon_ask, high_price.raw_epsilon_ask, atol=1e-12
    )

    gamma, k = maker.canonical_parameters(cfg.P0)
    canonical = AvellanedaStoikovMarketMaker(
        gamma=gamma,
        k=k,
        risk_horizon_steps=maker.risk_horizon_steps,
        cfg=cfg,
    )
    normalized_quote = maker.quote(4.0, cfg.P0)
    canonical_quote = canonical.quote(4.0, cfg.P0)
    assert np.isclose(
        normalized_quote.raw_epsilon_bid,
        canonical_quote.raw_epsilon_bid,
        atol=1e-12,
    )
    assert np.isclose(
        normalized_quote.raw_epsilon_ask,
        canonical_quote.raw_epsilon_ask,
        atol=1e-12,
    )

    assert maker.rho is not None and maker.kappa is not None
    print("A-S mathematical sanity passed")
    print(f"rho={maker.rho:.12g}")
    print(f"kappa={maker.kappa:.12g}")
    print(f"gamma(P0)={gamma:.12g}")
    print(f"k(P0)={k:.12g}")


if __name__ == "__main__":
    main()
