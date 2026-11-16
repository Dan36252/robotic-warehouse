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


class _RunningMeanStd:
    """Welford's online algorithm for a running mean/variance estimate."""

    def __init__(self):
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / self.count if self.count > 1 else 1.0


class CentralizedWarehouseEnv(gym.Wrapper):
    """Flattens rware's per-agent observation/action tuples into single joint
    vectors, turning the multi-agent warehouse into one large single-agent
    control problem: observation is the concatenation of every agent's
    observation, and action is a `MultiDiscrete` concatenating every agent's
    [movement, msg_bit, msg_bit, ...] sub-action. Reward is the mean of the
    per-agent rewards (identical under `RewardType.GLOBAL`).

    Optionally normalizes that reward by a running estimate of the standard
    deviation of the discounted return (the same approach as Stable-Baselines3's
    `VecNormalize(norm_reward=True)`), so the critic always sees rewards on a
    roughly consistent scale regardless of how large the environment's sparse
    pickup/delivery bonuses are relative to the dense shaping reward. Only the
    scale is adjusted -- the reward is never mean-shifted, since that would
    distort the sign of the incentive.
    """

    def __init__(
        self,
        env: Warehouse,
        normalize_reward: bool = True,
        reward_gamma: float = 0.99,
        reward_clip: float = 10.0,
        reward_epsilon: float = 1e-8,
    ):
        super().__init__(env)
        self.n_agents = env.n_agents

        self.normalize_reward = normalize_reward
        self.reward_gamma = reward_gamma
        self.reward_clip = reward_clip
        self.reward_epsilon = reward_epsilon
        self._discounted_return = 0.0
        self._return_rms = _RunningMeanStd()

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
        self._discounted_return = 0.0
        return np.concatenate(obs).astype(np.float32), info

    def step(self, action):
        n_sub_actions = len(self._per_agent_nvec)
        per_agent_actions = np.asarray(action).reshape(self.n_agents, n_sub_actions)
        obs, rewards, terminated, truncated, info = self.env.step(
            list(per_agent_actions)
        )
        flat_obs = np.concatenate(obs).astype(np.float32)
        reward = float(np.mean(rewards))

        if self.normalize_reward:
            self._discounted_return = (
                self._discounted_return * self.reward_gamma + reward
            )
            self._return_rms.update(self._discounted_return)
            reward = reward / np.sqrt(self._return_rms.var + self.reward_epsilon)
            if self.reward_clip is not None:
                reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
            if terminated or truncated:
                self._discounted_return = 0.0

        return flat_obs, reward, terminated, truncated, info


def make_centralized_env(
    *args,
    device=None,
    normalize_reward: bool = True,
    reward_gamma: float = 0.99,
    reward_clip: float = 10.0,
    **kwargs,
) -> tuple[GymWrapper, list[int]]:
    """Builds a `Warehouse` wrapped for centralized, TorchRL-based control.

    Positional/keyword arguments (other than the reward-normalization options
    above) are forwarded to `Warehouse.__init__`. Returns the TorchRL env
    together with the flattened action cardinalities (`nvec`), needed to build
    an `IndependentMultiCategorical` policy head.
    """
    env = Warehouse(*args, **kwargs)
    env = CentralizedWarehouseEnv(
        env,
        normalize_reward=normalize_reward,
        reward_gamma=reward_gamma,
        reward_clip=reward_clip,
    )
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
