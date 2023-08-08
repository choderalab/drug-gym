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
        library: Optional[MoleculeCollection] = None
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
                'molecules': Sequence(Discrete(self.max_molecules)),
                'num_analogs': Discrete(2), # 0: 1, 1: 5, 2: 96, 3: 384
                'fraction_random': Box(low=0.0, high=1.0, shape=(1,))
            }),
            'order': Dict({
                'assay': Discrete(len(self.assays)),
                'molecules': Sequence(Discrete(self.max_molecules))
            })
        })

        # Define the observation space
        self.observation_space = Dict({
            'library': Box(low=-float('inf'), high=float('inf'), shape=(self.max_molecules,)),  
            'order': Box(low=-float('inf'), high=float('inf'), shape=(10,))
        })

        # Initialize the library and orders
        if library is None:
            library = MoleculeCollection()
        self.library = library.clone()
        self.orders = []

        # Initialize the action mask
        self.valid_actions = np.zeros(self.max_molecules, dtype='int8')
        self.valid_actions[:len(self.library)] = True


    def step(self, action):

        # If the action includes a design, update library and action mask
        if 'design' in action:
            self.library += self.design_library(action['design'])
            self.valid_actions[:len(self.library)] = True

        # If the action includes an order, perform the order
        if 'order' in action:
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
        batch_sizes = {0: 1, 1: 10, 2: 96, 3: 384} # make more elegant?
        valid_indices = [m for m in action['molecules'] if self.valid_actions[m]]
        molecules = self.library[valid_indices]
        return self.library_designer.design(
            molecules,
            batch_sizes[action['num_analogs']],
            action['fraction_random']
        )

    def perform_order(self, action) -> None:

        # subset assay and molecules
        assay_index, molecule_indices = action['assay'], action['molecules']
        assay = self.assays[assay_index]

        valid_indices = [m for m in molecule_indices if self.valid_actions[m]]
        molecules = self.library[valid_indices]
        
        # perform inference
        results = assay(molecules)
        
        # update library annotations for molecules measured
        for idx, molecule in enumerate(molecules):
            molecule.update_annotations({assay.name: results[idx]})

    def get_observation(self):
        return self.library
        # return OrderedDict({'design': self.library, 'order': self.})

    def get_reward(self):
        # Implement the logic for calculating the reward based on the current state
        ...

    def check_done(self):
        # Implement the logic for checking if the episode is done
        ...

    def is_library_exhausted(self):
        # Implement the logic for checking if the library is exhausted
        ...
