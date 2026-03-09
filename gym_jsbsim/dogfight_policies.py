from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class OpponentPolicy:
    def reset(self) -> None:
        pass

    def action(self, observation: np.ndarray, action_space) -> np.ndarray:
        raise NotImplementedError


class StableOpponentPolicy(OpponentPolicy):
    """
    Neutral controls only.

    The aircraft still moves according to the existing JSBSim task setup and
    flight dynamics; this policy simply does not inject learned or scripted
    maneuvering.
    """

    def action(self, observation: np.ndarray, action_space) -> np.ndarray:
        return np.zeros(action_space.shape, dtype=np.float32)


class RandomOpponentPolicy(OpponentPolicy):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def action(self, observation: np.ndarray, action_space) -> np.ndarray:
        return self.rng.uniform(low=action_space.low, high=action_space.high).astype(np.float32)


@dataclass
class SB3PolicyOpponent(OpponentPolicy):
    model: Any
    deterministic: bool = True
    vecnormalize: Optional[Any] = None

    def action(self, observation: np.ndarray, action_space) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if self.vecnormalize is not None:
            obs = self.vecnormalize.normalize_obs(obs.reshape(1, -1))[0]
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


def load_sb3_policy_opponent(model_path: str,
                             observation_space,
                             action_space,
                             vecnormalize_path: Optional[str] = None,
                             deterministic: bool = True) -> SB3PolicyOpponent:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _ObsOnlyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = observation_space
            self.action_space = action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

    vec_env = DummyVecEnv([lambda: _ObsOnlyEnv()])
    vecnormalize = None
    if vecnormalize_path:
        vecnormalize = VecNormalize.load(vecnormalize_path, vec_env)
        vecnormalize.training = False
        vecnormalize.norm_reward = False
        env_for_model = vecnormalize
    else:
        env_for_model = vec_env
    model = PPO.load(model_path, env=env_for_model)
    return SB3PolicyOpponent(model=model, deterministic=deterministic, vecnormalize=vecnormalize)
