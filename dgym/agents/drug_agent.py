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
        
        # Extract action utility from the policy
        utility = self.policy(observations)
        
        # Apply negative bias to utility of masked actions (True = valid)
        if mask:
            utility[~mask] = -1e8

        # Compute probabilities from logits to get probabilities
        probs = self._softmax(utility)

        # Sample from action distribution
        molecules = np.random.choice(
            range(len(probs)),
            size=self.branch_factor,
            replace=False,
            p=probs
        ).tolist()

        # Construct actions
        action = self.construct_action(molecules)

        return action

    def policy(self, observations):
        raise NotImplementedError

    def construct_action(self, molecules):
        raise NotImplementedError

    @staticmethod
    def _softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=0, keepdims=True)



class HardcodedDrugAgent(DrugAgent):

    def __init__(
        self,
        action_space,
        scoring_functions: list,
        utility_function,
        num_analogs,
        branch_factor=5,
        # temperature=0.1,
        fraction_random = 0.1,
        epsilon=0.2,
    ) -> None:

        super().__init__(
            action_space,
            branch_factor = branch_factor,
            epsilon = epsilon,
        )

        self.scoring_functions = scoring_functions
        self.utility_function = utility_function
        self.num_analogs = num_analogs
        self.fraction_random = fraction_random
        # self.temperature = temperature

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
        # score library
        scores = [scorer(observations)
                  for scorer in self.scoring_functions]
        
        # convert scores to utility
        utility = [self.utility_function(score)
                   for score in zip(*scores)]
        
        return utility

    def construct_action(self, molecules):

        parameters = {}

        if self.action_type == 'ideate':
            parameters.update({
                'num_analogs': self.num_analogs,
                'fraction_random': self.fraction_random
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
