from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from config import Config, seed_everything
from networks import ActorCritic


def discount_cumsum(values: np.ndarray, discount: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + discount * running
        result[index] = running
    return result


class RolloutBuffer:
    def __init__(self, observation_dim: int, action_dim: int, size: int, cfg: Config) -> None:
        self.observations = np.zeros((size, observation_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.cfg = cfg
        self.pointer = 0
        self.path_start = 0

    def store(self, observation, action, reward, value, log_prob) -> None:
        index = self.pointer
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.values[index] = value
        self.log_probs[index] = log_prob
        self.pointer += 1

    def finish_path(self, last_value: float) -> None:
        path = slice(self.path_start, self.pointer)
        rewards = np.append(self.rewards[path], last_value)
        values = np.append(self.values[path], last_value)
        deltas = rewards[:-1] + self.cfg.gamma * values[1:] - values[:-1]
        self.advantages[path] = discount_cumsum(deltas, self.cfg.gamma * self.cfg.gae_lambda)
        self.returns[path] = discount_cumsum(rewards, self.cfg.gamma)[:-1]
        self.path_start = self.pointer

    def get(self) -> tuple[torch.Tensor, ...]:
        advantage_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - self.advantages.mean()) / advantage_std
        arrays = (self.observations, self.actions, self.advantages, self.returns, self.log_probs)
        self.pointer = self.path_start = 0
        return tuple(torch.as_tensor(array, device=self.cfg.device) for array in arrays)


@dataclass
class TrainingHistory:
    steps: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    evaluation: list[dict] = field(default_factory=list)


class PPOAgent:
    def __init__(self, env, cfg: Config | None = None) -> None:
        self.env = env
        self.cfg = cfg or Config()
        seed_everything(self.cfg.seed)
        observation_dim = env.reset().shape[0]
        self.network = ActorCritic(
            observation_dim, 3, self.cfg.hidden_size, self.cfg.hidden_layers
        ).to(self.cfg.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.lr)
        self.buffer = RolloutBuffer(observation_dim, 3, self.cfg.horizon, self.cfg)
        self.history = TrainingHistory()

    def select_action(self, observation: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.cfg.device)
        with torch.no_grad():
            distribution, value = self.network.distribution_and_value(obs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(-1)
        return action.cpu().numpy(), value.item(), log_prob.item()

    @torch.no_grad()
    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.cfg.device)
        return self.network(obs)[0].cpu().numpy()

    def train(self, evaluator=None, risk_penalty: bool = False) -> TrainingHistory:
        observation = self.env.reset()
        total_steps = 0
        rollout_reward = 0.0
        while total_steps < self.cfg.total_steps:
            steps_this_rollout = min(self.cfg.horizon, self.cfg.total_steps - total_steps)
            if steps_this_rollout != self.cfg.horizon:
                raise ValueError("total_steps must be a multiple of horizon")
            for _ in range(steps_this_rollout):
                action, value, log_prob = self.select_action(observation)
                next_observation, reward, _, info = self.env.step(action)
                if risk_penalty:
                    reward -= self.cfg.risk_penalty_alpha * info["inventory_pnl"] ** 2
                self.buffer.store(observation, action, reward, value, log_prob)
                observation = next_observation
                rollout_reward += reward
                total_steps += 1
            with torch.no_grad():
                obs = torch.as_tensor(observation, dtype=torch.float32, device=self.cfg.device)
                last_value = self.network.distribution_and_value(obs)[1].item()
            self.buffer.finish_path(last_value)
            self._update()
            self.history.steps.append(total_steps)
            self.history.rewards.append(rollout_reward)
            rollout_reward = 0.0
            if evaluator and (total_steps % self.cfg.eval_every_steps == 0 or total_steps == self.cfg.total_steps):
                metrics = evaluator(self)
                self.history.evaluation.append({"step": total_steps, **metrics})
                print(f"step {total_steps:>7}: mean PnL {metrics['mean_total_pnl']:.3f}")
        return self.history

    def _update(self) -> None:
        observations, actions, advantages, returns, old_log_probs = self.buffer.get()
        for _ in range(self.cfg.n_epochs):
            indices = np.random.permutation(self.cfg.horizon)
            for start in range(0, self.cfg.horizon, self.cfg.minibatch_size):
                batch = indices[start : start + self.cfg.minibatch_size]
                distribution, values = self.network.distribution_and_value(observations[batch])
                log_probs = distribution.log_prob(actions[batch]).sum(-1)
                ratio = (log_probs - old_log_probs[batch]).exp()
                objective = ratio * advantages[batch]
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages[batch]
                policy_loss = -torch.min(objective, clipped).mean()
                value_loss = (returns[batch] - values).square().mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
