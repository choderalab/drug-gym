import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable

class DrugAgent:

    def __init__(
        self,
        action_space,
        branch_factor=5,
        epsilon=0.1,
    ) -> None:

        self.action_space = action_space
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

        return action

    def policy(self, observations):
        raise NotImplementedError

    def construct_action(self, molecules):
        raise NotImplementedError


class SequentialDrugAgent(DrugAgent):

    def __init__(
        self,
        action_space,
        utility_function,
        num_analogs,
        branch_factor=5,
        temperature=0.1,
        epsilon=0.2,
    ) -> None:

        super().__init__(
            action_space,
            branch_factor = branch_factor,
            epsilon = epsilon,
        )

        self.utility_function = utility_function
        self.num_analogs = num_analogs
        self.temperature = temperature

        # action utils
        self._encoder = {
            'ideate': 0,
            'prioritize': 1
        }

        self._transitions = {
            'ideate': 'prioritize',
            'prioritize': 'ideate'
        }

        self.action_type = 'ideate'


    def policy(self, observations):
        """
        """
        # convert scores to utility
        return self.utility_function(observations)

    def construct_action(self, molecules):

        parameters = {}

        if self.action_type == 'ideate':
            parameters.update({
                'num_analogs': self.num_analogs,
                'temperature': self.temperature
            })
        
        action = {
            'type': self._encoder[self.action_type],
            'molecules': molecules,
            'parameters': parameters
        }

        # iterate action type
        self.action_type = self._transitions[self.action_type]
        
        return action

    def reset(self):
        self.action_state = 'ideate'

    def learn(self, previous_observation, action, reward, observation, done):
        """Implement your learning algorithm here"""
        pass
