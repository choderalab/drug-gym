import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable
import itertools

class DrugAgent:

    def __init__(
        self,
        branch_factor=5,
        epsilon=0.1,
    ) -> None:

        self.branch_factor = branch_factor
        self.epsilon = epsilon

    def act(self, observations, mask=None):

        # check index error
        branches = min([self.branch_factor, len(observations)])

        # Extract action utility from the policy
        utility = self.policy(observations)
        
        # Apply negative bias to utility of masked actions (True = valid)
        if mask:
            utility[~mask] = -1e8

        # Epsilon greedy selection
        molecules = []
        sorted_utility = np.argsort(utility)[::-1]
        for i in range(branches):
            if np.random.rand() < self.epsilon:
                molecules.append(np.random.choice(len(utility)))
            else:
                molecules.append(sorted_utility[i])

        # Construct action
        action = self.construct_action(molecules)
        action.setdefault('parameters', {})
        action.update({'molecules': molecules})

        return action

    def policy(self, observations):
        raise NotImplementedError

    def construct_action(self, molecules):
        raise NotImplementedError


class SequentialDrugAgent(DrugAgent):

    def __init__(
        self,
        sequence,
        utility_function,
        branch_factor=5,
        epsilon=0.2,
    ) -> None:

        super().__init__(
            branch_factor = branch_factor,
            epsilon = epsilon,
        )

        self.utility_function = utility_function
        self.sequence = sequence
        self._iter_sequence = itertools.cycle(sequence)

    def policy(self, observations):
        """
        """
        # convert scores to utility
        return self.utility_function(observations)

    def construct_action(self, molecules):
        return next(self._iter_sequence)

    def learn(self, previous_observation, action, reward, observation, done):
        """Implement your learning algorithm here"""
        pass
