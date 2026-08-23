from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass
class Config:
    # Market
    P0: float = 100.0
    mu: float = 0.0
    sigma: float = 0.2
    dt: float = 1.0 / (252 * 6.5 * 60 / 15)
    relative_price: bool = False
    include_fill_feedback: bool = False
    paper_pnl_observation: bool = False
    observation_mode: str = "default"
    num_investors: int = 20
    order_size_mode: str = "gamma"
    investor_shape: float = 2.0
    investor_scale: float = 1.0
    buy_probability: float = 0.5
    base_spread_bp: float = 2.0
    lob_slope_bp: float = 0.2

    # PPO
    rollout: int = 20
    horizon: int = 1024
    n_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.95
    gae_value_target: bool = False
    clip_eps: float = 0.2
    lr: float = 5e-5
    vf_coef: float = 0.5
    vf_clip_param: float | None = None
    ent_coef: float = 3e-3
    kl_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    hidden_layers: int = 2
    fixed_policy_std: float | None = None

    # Experiment
    seed: int = 42
    device: str = "cpu"
    eval_every_steps: int = 4096
    eval_episodes: int = 10
    eval_days: int = 20
    risk_penalty_alpha: float = 0.01

    @property
    def total_steps(self) -> int:
        return self.rollout * self.horizon

    @property
    def steps_per_day(self) -> int:
        return int(6.5 * 60 / 15)

    def updated(self, **changes: object) -> "Config":
        return replace(self, **changes)

    def __post_init__(self) -> None:
        if self.num_investors <= 0:
            raise ValueError("num_investors must be positive")
        if self.order_size_mode not in {"gamma", "unit"}:
            raise ValueError("order_size_mode must be 'gamma' or 'unit'")
        if self.observation_mode not in {"default", "paper"}:
            raise ValueError("observation_mode must be 'default' or 'paper'")
        if not 0.0 <= self.buy_probability <= 1.0:
            raise ValueError("buy_probability must lie in [0, 1]")
        if self.investor_shape <= 0.0 or self.investor_scale <= 0.0:
            raise ValueError("Gamma order-size parameters must be positive")
        if self.kl_coef < 0.0:
            raise ValueError("kl_coef must be nonnegative")
        if self.vf_clip_param is not None and self.vf_clip_param <= 0.0:
            raise ValueError("vf_clip_param must be positive when set")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
