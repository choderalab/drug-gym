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
        library: Optional[MoleculeCollection] = MoleculeCollection()
    ) -> None:
        
        super().__init__()
        
        self.library_designer = library_designer

        # Define the maximum number of molecules that could ever be synthesized
        self.max_molecules = budget

        # Define assays
        self.assays = assays

        # Define the action space
        self.action_space = Dict({
            'design': Dict({
                'selected_molecules': Sequence(Discrete(self.max_molecules)),
                'num_analogs': Discrete(self.max_molecules),
                'fraction_random': Box(low=0.0, high=1.0, shape=(1,))
            }),
            'order': Dict({
                'assay': Discrete(len(self.assays)),
                'selected_molecules': Sequence(Discrete(self.max_molecules))
            })
        })

        # Define the observation space
        self.observation_space = Dict({
            'library': Box(low=-float('inf'), high=float('inf'), shape=(self.max_molecules,)),  
            'order': Box(low=-float('inf'), high=float('inf'), shape=(10,))
        })

        # Initialize the library and orders
        self.library = library
        self.orders = []

        # Initialize the action mask
        self.action_mask = np.zeros(self.max_molecules, dtype=bool)


    def step(self, action):
        
        # If the action includes a design, update library and action mask
        if 'design' in action:
            self.library += self.design_library(action['design'])
            self.action_mask[len(self.library):] = True

        # If the action includes an order, perform the order
        if 'order' in action:
            molecule_index = action['order']['molecule']
            # if not self.action_mask[molecule_index]:
            #     raise ValueError(f"The action for molecule {molecule_index} is masked.")
            self.orders.append(self.perform_order(action['order']))

        # Calculate the reward and check if the episode is done
        reward = self.get_reward()
        done = self.check_done()

        return self.get_observation(), reward, done, {}

    def reset(self):
        self.library = None
        self.orders = []
        self.current_stage = 'design'
        return self.get_observation()

    def design_library(self, action):
        """
        Returns the 
        """
        selected_molecules = self.library[action['selected_molecules']]
        return self.library_designer.design(
            selected_molecules,
            action['num_analogs'],
            action['percent_random']
        )

    def perform_order(self, action):

        results = []
        selected_assay = self.assays[action['assay']]
        selected_molecules = self.library[action['selected_molecules']]
        result = selected_assay(selected_molecules)


    def get_observation(self):
        # Implement the logic for generating the observation based on the current state
        return self.library

    def get_reward(self):
        # Implement the logic for calculating the reward based on the current state
        ...

    def check_done(self):
        # Implement the logic for checking if the episode is done
        ...

    def is_library_exhausted(self):
        # Implement the logic for checking if the library is exhausted
        ...
