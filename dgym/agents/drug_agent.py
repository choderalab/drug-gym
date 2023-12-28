import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable
import itertools
import dgym as dg

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
        # TODO - only choose among latest cycle molecules?
        if action['name'] == 'ideate':

            # Only annotated molecules
            observations = observations.annotated

            # Only latest-cycle
            current_cycle = max(o.design_cycle for o in observations)
            observations = [o for o in observations if o.design_cycle == current_cycle]

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

class MultiStepDrugAgent(SequentialDrugAgent):

    def __init__(
        self,
        num_steps,
        designer,
        agg_func = np.mean,
        *args,
        **kwargs
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.num_steps = num_steps
        self.designer = designer
        self.agg_func = agg_func

    def policy(self, observations):
        """
        """
        # Only lookahead if ideating
        if any(not obs.annotations for obs in observations) \
            or self.num_steps < 2:
            return self.utility_function(observations)

        aggregated_scores = []
        for obs in observations:
            all_scores = self.multi_step_lookahead(obs)
            aggregated_score = self.agg_func(all_scores)
            aggregated_scores.append(aggregated_score)
        
        return aggregated_scores
    
    def multi_step_lookahead(self, molecule):
        """
        Perform a multi-step lookahead to explore molecule analogs.

        Parameters
        ----------
        molecule : Molecule
            The starting molecule for the design process.
        depth : int
            The number of steps to look ahead.

        Returns
        -------
        list of float
            A list of evaluation scores from all analogs generated up to the given depth.
        """
        # Starting with the initial molecule
        molecules = [molecule]

        # Explore for each depth level
        for _ in range(self.num_steps - 1):
            new_molecules = []
            for molecule in molecules:
                new_molecules.extend(self.designer.design(molecule, 10))
            molecules = new_molecules

        # Evaluate all molecules and return their scores
        return self.utility_function(molecules)
