import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable


class DrugAgent:

    def __init__(self, action_space, utiity_function):

        self.action_space = action_space
        self.mode = 'ideate'
        self.scoring_function = utility_function.score

    def act(self, observations, mask):
        
        # Extract action values or logits from the policy
        action_values = self.policy(observations)
        
        # Apply large negative bias to action values of masked actions (True = valid)
        action_values[~mask] = -1e8

        # Softmax logits to get probabilities
        action_probs = softmax(action_values)

        # Sample from action distribution
        action = np.random.choice(range(len(action_probs)), p=action_probs)

        return action

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
        if self.mode == 'ideate':

            # ideate
            action = {
                'design': {
                    'molecules': chosen_molecules,
                    'num_analogs': 1,
                    'fraction_random': 0.0
                }
            }
            
            self.mode = 'triage'

        elif self.mode == 'triage':

            # score library
            library = observations[0]
            assay_results = [assay(library) for assay in assays]
            utility = [self.scoring_function(properties)
                       for properties in zip(*assay_results)]

            # triage
            chosen_molecules = utility.argsort()[-5:].tolist()
            if len(library) > 2:
                chosen_molecules.extend(
                    random.sample(range(len(library)), 2)
                )

            action = [
                {'order': {'assay': 0, 'molecules': chosen_molecules}},
                {'order': {'assay': 0, 'molecules': chosen_molecules}}
            ]

            self.mode = 'ideate'

        return action

    def learn(self, previous_observation, action, reward, observation, done):
        """Implement your learning algorithm here"""
        pass
