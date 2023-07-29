import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable


class DrugAgent:

    def __init__(self, action_space):

        self.action_space = action_space


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

        """
        


    def learn(self, previous_observation, action, reward, observation, done):
        """Implement your learning algorithm here"""
        pass
