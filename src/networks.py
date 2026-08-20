from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int = 3, hidden_size: int = 256, hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = observation_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_size), nn.Tanh()])
            width = hidden_size
        self.shared = nn.Sequential(*layers)
        self.policy_mean = nn.Linear(width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(width, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(observations)
        raw_mean = self.policy_mean(hidden)
        mean = torch.cat([torch.tanh(raw_mean[..., :2]), torch.sigmoid(raw_mean[..., 2:])], dim=-1)
        return mean, self.value_head(hidden).squeeze(-1)

    def distribution_and_value(self, observations: torch.Tensor):
        mean, value = self(observations)
        distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return distribution, value

    # TODO(replication pass): replace Gaussian sampling plus environment clipping
    # with a mathematically consistent bounded continuous-action distribution.
