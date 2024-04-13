import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable
import itertools
import dgym as dg
from .exploration import ExplorationStrategy

class DrugAgent:

    def __init__(
        self,
        utility_function: Callable,
        exploration_strategy: ExplorationStrategy,
    ) -> None:

        self.utility_function = utility_function
        self.exploration_strategy = exploration_strategy

    def act(self, observations, mask=None):

        # Construct action
        action = self.construct_action()
        
        # Filter observations by action
        match action['name']:
            case 'design':
                observations = observations.tested or observations
            case 'make':
                observations = observations.designed
            case _ as test:
                if 'Noisy' in test:
                    observations = observations.designed
                else:
                    observations = observations.made

        # Compute utility from the policy
        utility = self.policy(observations)
        
        # Apply negative bias to utility of masked actions
        if mask:
            utility[~mask] = -1e8

        # Gather molecule indices
        batch_size = min([action.pop('batch_size'), len(observations)])
        pending = self.exploration_strategy(utility, size=batch_size)
        molecules = [observations.index[p] for p in pending]

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
        return self.utility_function(
            observations,
            method='hybrid',
            use_precomputed=True,
        )

    def construct_action(self):
        return next(self._iter_sequence).copy()
    
    def reset(self):
        self._iter_sequence = itertools.cycle(self.sequence)


class LearningDrugAgent(DrugAgent):
    
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def learn(
        self,
        previous_observation,
        action,
        reward,
        observation,
        done
    ):
        pass


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
        if any(not o.annotations for o in observations) \
            or self.num_steps < 2:
            return self.utility_function(observations)

        leaves = []
        for observation in observations:
            leaves_ = self.multi_step_lookahead(observation)
            leaves.extend(leaves_)
        
        scores = self.utility_function(leaves).reshape(len(observations), -1)
        aggregated_scores = self.agg_func(scores, axis=1)
        
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

        return molecules


class MCTSDrugAgent(MultiStepDrugAgent):

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.nodes = dict()

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
