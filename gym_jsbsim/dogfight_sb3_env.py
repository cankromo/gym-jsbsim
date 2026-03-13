from typing import Optional, Sequence

import gym
import numpy as np

from gym_jsbsim.dogfight import DogfightEnv
from gym_jsbsim.dogfight_policies import OpponentPolicy, StableOpponentPolicy


class DogfightSingleAgentEnv(gym.Env):
    """
    SB3-facing wrapper around the additive two-plane dogfight env.

    One aircraft is controlled by SB3. The opponent uses a separate policy
    object, but both aircraft still step through the same underlying JSBSim
    world and task logic.
    """

    metadata = {"render.modes": []}

    def __init__(self,
                 controlled_agent: str = "plane_a",
                 opponent_policy: Optional[OpponentPolicy] = None,
                 scenario_names: Optional[Sequence[str]] = None):
        super().__init__()
        self.core = DogfightEnv(scenario_names=scenario_names)
        if controlled_agent not in self.core.agent_order:
            raise ValueError(f"Unknown controlled agent: {controlled_agent}")
        self.controlled_agent = controlled_agent
        self.opponent_agent = self.core._opponent(controlled_agent)
        self.opponent_policy = opponent_policy or StableOpponentPolicy()
        self.observation_space = self.core.observation_spaces[self.controlled_agent]
        self.action_space = self.core.action_spaces[self.controlled_agent]
        self._last_obs = None

    def reset(self):
        self.opponent_policy.reset()
        obs = self.core.reset()
        self._last_obs = obs
        return np.asarray(obs[self.controlled_agent], dtype=np.float32)

    def step(self, action):
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before step().")

        opponent_obs = np.asarray(self._last_obs[self.opponent_agent], dtype=np.float32)
        opponent_action = self.opponent_policy.action(
            observation=opponent_obs,
            action_space=self.core.action_spaces[self.opponent_agent],
        )
        obs, rewards, dones, infos = self.core.step({
            self.controlled_agent: np.asarray(action, dtype=np.float32),
            self.opponent_agent: np.asarray(opponent_action, dtype=np.float32),
        })
        self._last_obs = obs
        done = bool(dones.get(self.controlled_agent, False) or dones.get("__all__", False))
        info = dict(infos.get(self.controlled_agent, {}))
        info["opponent_reward"] = float(rewards[self.opponent_agent])
        info["opponent_done"] = bool(dones.get(self.opponent_agent, False))
        return (
            np.asarray(obs[self.controlled_agent], dtype=np.float32),
            float(rewards[self.controlled_agent]),
            done,
            info,
        )

    def close(self):
        self.core.close()
