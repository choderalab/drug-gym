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
        scoring_function=None,
        branch_factor=5,
        temperature=0.1,
    ) -> None:

        self.action_space = action_space
        self.scoring_function = scoring_function

    def act(self, observations, mask=None):
        
        # Extract action utility from the policy
        utility = self.policy(observations)
        
        # Apply negative bias to utility of masked actions (True = valid)
        if mask:
            utility[~mask] = -1e8

        # Compute probabilities from logits to get probabilities
        probs = self.boltzmann(utility)

        # Sample from action distribution
        molecules = np.random.choice(
            range(len(probs)),
            size=self.branch_factor,
            replace=False,
            p=probs
        )

        # Construct actions
        action = self.construct_action(molecules)

        return action

    def policy(self, observations):
        raise NotImplementedError

    def construct_action(self, molecules):
        raise NotImplementedError

    @staticmethod
    def boltzmann(utility, temperature):
        """Compute Boltzmann probabilities for a given set of utilities and temperature."""
        energies = -np.array(utility)
        exp_energies = np.exp(-energies / temperature)
        return exp_energies / np.sum(exp_energies)


class HardcodedDrugAgent(DrugAgent):

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
            utility_function.score,
            # branch_factor,
            # temperature,
        )
        self.mode = 'ideate'
        self.transitions = {
            'ideate': 'prioritize',
            'prioritize': 'ideate'
        }

    def policy(self, observations):
        """
        Parts of policy:
            1. scoring
            2. decision criterion

        The loop according to John:
            1. Ideate
            2. Score
            3. Triage
            4. Assay
            5. Update Models

        """
        # score library
        library = observations[0]
        assay_results = [assay(library) for assay in assays]
        utility = [self.scoring_function(properties)
                   for properties in zip(*assay_results)]
        return utility

    def construct_action(self, molecules):
        action = self.__getattr__(self.action_state)(molecules)
        self.mode = self.transitions[self.mode]
        return action

    def ideate(self, molecules):
        # ideate
        action = {
            'design': {
                'molecules': molecules,
                'num_analogs': self.num_analogs,
                'fraction_random': 0.0
            }
        }
        return action

    def prioritize(self, molecules):
        action = [
            {'order': {'assay': 0, 'molecules': molecules}},
            {'order': {'assay': 1, 'molecules': molecules}}
        ]
        return action

    def reset(self):
        self.action_state = 'ideate'

    def learn(self, previous_observation, action, reward, observation, done):
        """Implement your learning algorithm here"""
        pass
