from __future__ import annotations

import rdkit
from typing import Iterable, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import (
    Dict, Discrete, Box, Sequence, Tuple
)

from dgym.collection import MoleculeCollection

class DrugEnv(gym.Env):
    """
    ## Description
    This is a ...

    To solve ..., you need to (achieve reward) in (budget).
    
    ## Action Space
    Actions are (values) in (space) and represent.
    
    ## Observation Space
    State consists of ...
    
    ## Rewards
    Reward is given for (...).
    If (...), it gets (negative number canceling out). (Actions) cost a small
    amount of (cost). A more optimal agent will get a better score.
    
    ## Starting State
    The (env) starts (...).
    
    ## Episode Termination
    The episode will terminate if (...) or (...).
    
    ## Arguments
    Extra instructions about use of the environment:
    ```python
    import gymnasium as gym
    env = gym.make("DrugEnv")
    ```
    """

    def __init__(
        self,
        library_designer,
        budget: int = 10_000,
        assays: list = [],
        library: Optional[MoleculeCollection] = None,
        utility_function: Optional[UtilityFunction] = None,
    ) -> None:
        
        super().__init__()
        
        self.library_designer = library_designer

        self.budget = budget

        # For now, max library size is equivalent to budget
        self.max_molecules = self.budget

        # Define assays
        self.assays = assays

        # Define utility function
        self.utility_function = utility_function

        # Define the action space
        self.action_space = Dict({
            'type': Discrete(len(self.assays)),
            'molecules': Sequence(Discrete(self.max_molecules)),
            # 'parameters': None, # TBD
        })

        # Define the observation space
        self.observation_space = Dict({
            'library': Box(low=-float('inf'), high=float('inf'), shape=(self.max_molecules,)),  
            'order': Box(low=-float('inf'), high=float('inf'), shape=(10,))
        })

        # Initialize the library and orders
        if library is None:
            library = MoleculeCollection()
        
        self._library_0 = library.clone()
        self.library = self._library_0.clone()
        self.reward_history = []

        # Initialize the action mask
        # TODO - figure out the logic for the case when the budget is smaller than the initial library
        self.valid_actions = np.zeros(self.max_molecules, dtype='int8')
        self.valid_actions[:len(self.library)] = True

        # Keep track of design cycles
        self.design_cycle = 0


    def step(self, action):
        """
        Run one timestep of the environment’s dynamics using the agent actions.
        When the end of an episode is reached (terminated or truncated),
        it is necessary to call reset() to reset this environment’s state for the next episode.

        Parameters
        ----------
        action: ActType
            An action provided by the agent to update the environment state.

        Returns
        -------
        observation : ObsType
            An element of the environment’s observation_space as the next observation due to the agent actions.
            An example is a numpy array containing the positions and velocities of the pole in CartPole.

        reward: float
            The reward as a result of taking the action.

        terminated: bool
            Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative.
            An example is reaching the goal state or moving into the lava from the Sutton and Barton, Gridworld.
            If true, the user needs to call reset().

        truncated: bool
            Whether the truncation condition outside the scope of the MDP is satisfied.
            Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
            Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().

        info: dict
            Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            This might, for instance, contain: metrics that describe the agent’s performance state,
            variables that are hidden from observations, or individual reward terms that are combined
            to produce the total reward.

        """
        action_type, molecules, parameters = action.values()
        
        # Perform action
        if action_type == 0:
            self.library += self.design_library(molecules, **parameters)
        elif action_type >= 1:
            self.perform_order(action_type - 1, molecules, *parameters)

        # Update valid actions
        self.valid_actions[:len(self.library)] = True
        
        # Calculate the reward
        reward = self.get_reward()
        self.reward_history.append(reward)
        
        # Check if the episode is done
        terminated = self.check_terminated()
        truncated = self.check_truncated()

        return self.get_observation(), reward, terminated, truncated, {}

    def design_library(self, molecule_indices, num_analogs, temperature):
        """
        Returns the library of molecules.
        """
        batch_sizes = {0: 1, 1: 10, 2: 96, 3: 384} # make more elegant?
        
        # subset valid molecules
        valid_indices = [m for m in molecule_indices if self.valid_actions[m]]
        molecules = self.library[valid_indices]
        
        # design new library
        new_molecules = MoleculeCollection()
        for molecule in molecules:
            new_molecules += self.library_designer.design(
                molecule,
                size=batch_sizes[num_analogs],
                mode='analog',
                temperature=temperature
            )

        return new_molecules

    def perform_order(self, assay_index, molecule_indices, **params) -> None:

        # subset assay and molecules
        assay = self.assays[assay_index]
        valid_indices = [m for m in molecule_indices if self.valid_actions[m]]
        molecules = self.library[valid_indices]
        
        # perform inference
        results = assay(molecules, **params)
        
        # update library annotations for molecules measured
        molecules.update_annotations([{assay.name: r} for r in results])

    def get_observation(self):
        return self.library

    def get_reward(self):
        utility = self.utility_function(self.library)
        return max(utility)

    def check_terminated(self):
        # Implement the logic for checking if the episode is done
        return self.reward_history[-1] == 1

    def check_truncated(self):
        # Implement the logic for checking if the episode is done
        return len(self.library) >= self.budget

    def reset(self):
        self.design_cycle = 0
        self.library = self._library_0.clone()
        return self.get_observation(), {}