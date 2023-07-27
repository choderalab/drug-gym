import rdkit
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DrugEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, arg1, arg2, ...):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return observation, info

    def render(self):
        ...

    def close(self):
        ...


# class DrugEnv(gym.Env):
#     """Environment for Drug Discovery
    
#     Class that implements a gym.Env interface for drug discovery.
#     See https://gymnasium.farama.org/ for details on gymnasium.
    
#     """
#     def __init__(
#         self,
#     ) -> None:

#         # spaces
#         self.action_space = gym.spaces.Discrete(len(Actions))
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

#     def reset(self):
#         self._done = False
#         self.history = {}
#         return self._get_observation()
    
#     def step(self, action: Iterable[rdkit.Chem.Mol]):
#         self._done = False
#         self._current_tick += 1

#         if self._current_tick == self._end_tick:
#             self._done = True

#         step_reward = self._calculate_reward(action)
#         self._total_reward += step_reward

#         self._update_profit(action)

#         trade = False
#         if ((action == Actions.Buy.value and self._position == Positions.Short) or
#             (action == Actions.Sell.value and self._position == Positions.Long)):
#             trade = True

#         if trade:
#             self._position = self._position.opposite()
#             self._last_trade_tick = self._current_tick

#         self._position_history.append(self._position)
#         observation = self._get_observation()
#         info = dict(
#             total_reward = self._total_reward,
#             total_profit = self._total_profit,
#             position = self._position.value
#         )
#         self._update_history(info)

#         return observation, step_reward, self._done, info

    
# 	def _get_observation(self):
#         return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]


#     def _update_history(self, info):
#         if not self.history:
#             self.history = {key: [] for key in info.keys()}

#         for key, value in info.items():
#             self.history[key].append(value)


#     def _process_data(self):
#         raise NotImplementedError


#     def _calculate_reward(self, action):
#         raise NotImplementedError


#     def _update_profit(self, action):
#         raise NotImplementedError
