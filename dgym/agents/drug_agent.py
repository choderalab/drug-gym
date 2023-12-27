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
        utility_function,
        branch_factor=5,
        epsilon=0.1,
    ) -> None:

        self.utility_function = utility_function
        self.branch_factor = branch_factor
        self.epsilon = epsilon

    def act(self, observations, mask=None):

        # Construct action
        action = self.construct_action()
        
        # When ideating, only choose among annotated molecules
        if action['name'] == 'ideation':
            observations = observations.annotated

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

        # Add molecules to action
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
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        for seq in sequence:
            seq.setdefault('parameters', {})
        self.sequence = sequence
        self._iter_sequence = itertools.cycle(sequence)

    def policy(self, observations):
        """
        """
        # convert scores to utility
        return self.utility_function(observations)

    def construct_action(self):
        return next(self._iter_sequence).copy()

    def learn(self, previous_observation, action, reward, observation, done):
        """Implement your learning algorithm here"""
        pass

class NoisySequentialDrugAgent(SequentialDrugAgent):

    def __init__(
        self, noise, *args, **kwargs
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.noise = noise
    
    def policy(self, observations):
        utility = self.utility_function(observations)
        utility += np.random.normal(0, self.noise, len(utility))
        return utility