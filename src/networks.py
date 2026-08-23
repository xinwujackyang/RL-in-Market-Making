from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SquashedNormal:
    """Independent Normal transformed to [-1, 1], [-1, 1], and [0, 1]."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.base = torch.distributions.Normal(mean, std)

    @staticmethod
    def _transform(latent: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(latent)
        return torch.cat([squashed[..., :2], (squashed[..., 2:] + 1.0) / 2.0], dim=-1)

    @staticmethod
    def _inverse(action: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(action.dtype).eps
        unit_action = torch.cat([action[..., :2], 2.0 * action[..., 2:] - 1.0], dim=-1)
        unit_action = unit_action.clamp(-1.0 + eps, 1.0 - eps)
        return torch.atanh(unit_action)

    def sample(self) -> torch.Tensor:
        return self._transform(self.base.sample())

    def rsample(self) -> torch.Tensor:
        return self._transform(self.base.rsample())

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        latent = self._inverse(action)
        # Stable log(1 - tanh(u)^2). The hedge transform has another
        # derivative factor of 1/2, hence -log(2) in its log-Jacobian.
        log_tanh_jacobian = 2.0 * (torch.log(torch.tensor(2.0, device=latent.device)) - latent - F.softplus(-2.0 * latent))
        log_jacobian = log_tanh_jacobian.clone()
        log_jacobian[..., 2] -= torch.log(torch.tensor(2.0, device=latent.device))
        return self.base.log_prob(latent) - log_jacobian

    def entropy(self) -> torch.Tensor:
        """Base-Gaussian entropy proxy; transformed entropy has no simple form."""
        return self.base.entropy()


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 3,
        hidden_size: int = 256,
        hidden_layers: int = 2,
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        actor_layers: list[nn.Module] = []
        width = observation_dim
        for _ in range(hidden_layers):
            actor_layers.extend([nn.Linear(width, hidden_size), nn.Tanh()])
            width = hidden_size
        self.actor = nn.Sequential(*actor_layers)
        self.policy_mean = nn.Linear(width, action_dim)
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.policy_log_std = nn.Linear(width, action_dim)
            nn.init.zeros_(self.policy_log_std.weight)
            nn.init.zeros_(self.policy_log_std.bias)
            self.register_parameter("log_std", None)
        else:
            self.policy_log_std = None
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        critic_layers: list[nn.Module] = []
        critic_width = observation_dim
        for _ in range(hidden_layers):
            critic_layers.extend([nn.Linear(critic_width, hidden_size), nn.Tanh()])
            critic_width = hidden_size
        self.critic = nn.Sequential(*critic_layers)
        self.value_head = nn.Linear(critic_width, 1)

    def latent_policy_parameters(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_hidden = self.actor(observations)
        mean = self.policy_mean(actor_hidden)
        if self.policy_log_std is not None:
            log_std = self.policy_log_std(actor_hidden)
        else:
            log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def latent_mean_and_value(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _ = self.latent_policy_parameters(observations)
        critic_hidden = self.critic(observations)
        return mean, self.value_head(critic_hidden).squeeze(-1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_mean, value = self.latent_mean_and_value(observations)
        return SquashedNormal._transform(latent_mean), value

    def distribution_and_value(self, observations: torch.Tensor):
        mean, log_std = self.latent_policy_parameters(observations)
        critic_hidden = self.critic(observations)
        value = self.value_head(critic_hidden).squeeze(-1)
        distribution = SquashedNormal(mean, log_std.exp())
        return distribution, value
