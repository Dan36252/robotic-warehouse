"""TorchRL adapter for rware's Warehouse for training a single centralized
policy: one network that reads every agent's observation concatenated into
one vector, and outputs every agent's action concatenated into one vector.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical, Distribution

from rware.warehouse import Warehouse
from torchrl.envs.libs.gym import GymWrapper


class CentralizedWarehouseEnv(gym.Wrapper):
    """Flattens rware's per-agent observation/action tuples into single joint
    vectors, turning the multi-agent warehouse into one large single-agent
    control problem: observation is the concatenation of every agent's
    observation, and action is a `MultiDiscrete` concatenating every agent's
    [movement, msg_bit, msg_bit, ...] sub-action. Reward is the mean of the
    per-agent rewards (identical under `RewardType.GLOBAL`).
    """

    def __init__(self, env: Warehouse):
        super().__init__(env)
        self.n_agents = env.n_agents

        sa_action_space = env.action_space[0]
        if isinstance(sa_action_space, gym.spaces.Discrete):
            self._per_agent_nvec = np.array([sa_action_space.n], dtype=np.int64)
        else:
            self._per_agent_nvec = np.asarray(sa_action_space.nvec, dtype=np.int64)

        per_agent_obs_space = env.observation_space[0]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([per_agent_obs_space.low] * self.n_agents),
            high=np.concatenate([per_agent_obs_space.high] * self.n_agents),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.tile(self._per_agent_nvec, self.n_agents)
        )

    @property
    def nvec(self) -> list[int]:
        """Flattened per-sub-action cardinalities, e.g. [5,2,2,2]*n_agents."""
        return self.action_space.nvec.tolist()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate(obs).astype(np.float32), info

    def step(self, action):
        n_sub_actions = len(self._per_agent_nvec)
        per_agent_actions = np.asarray(action).reshape(self.n_agents, n_sub_actions)
        obs, rewards, terminated, truncated, info = self.env.step(
            list(per_agent_actions)
        )
        flat_obs = np.concatenate(obs).astype(np.float32)
        reward = float(np.mean(rewards))
        return flat_obs, reward, terminated, truncated, info


def make_centralized_env(*args, device=None, **kwargs) -> tuple[GymWrapper, list[int]]:
    """Builds a `Warehouse` wrapped for centralized, TorchRL-based control.

    Positional/keyword arguments are forwarded to `Warehouse.__init__`.
    Returns the TorchRL env together with the flattened action cardinalities
    (`nvec`), needed to build an `IndependentMultiCategorical` policy head.
    """
    env = Warehouse(*args, **kwargs)
    env = CentralizedWarehouseEnv(env)
    nvec = env.nvec
    return GymWrapper(env, categorical_action_encoding=True, device=device), nvec


class IndependentMultiCategorical(Distribution):
    """Joint distribution over a `MultiDiscrete` action made of independent
    categorical sub-actions with (possibly) different cardinalities.

    Needed because each rware agent's action is
    [movement(5-way), msg_bit(2-way), msg_bit(2-way), msg_bit(2-way)], and the
    centralized policy's action is the concatenation of that across every
    agent -- there's no off-the-shelf torch/torchrl distribution for a
    mixed-cardinality `MultiDiscrete` action space.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(self, logits: torch.Tensor, nvec: list[int], validate_args=None):
        self.nvec = [int(n) for n in nvec]
        self._cats = [
            Categorical(logits=chunk)
            for chunk in torch.split(logits, self.nvec, dim=-1)
        ]
        super().__init__(batch_shape=logits.shape[:-1], validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([c.sample(sample_shape) for c in self._cats], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        per_action = torch.unbind(value.long(), dim=-1)
        return torch.stack(
            [c.log_prob(v) for c, v in zip(self._cats, per_action)], dim=-1
        ).sum(-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([c.entropy() for c in self._cats], dim=-1).sum(-1)

    @property
    def deterministic_sample(self) -> torch.Tensor:
        return torch.stack([c.probs.argmax(-1) for c in self._cats], dim=-1)
