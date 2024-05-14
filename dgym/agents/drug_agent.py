import numpy as np
from typing import Iterable
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
        
        # Filter observations
        observations = self._filter_observations(observations, action)

        # Compute utility from the policy
        utility = self.policy(observations)
        
        # Apply negative bias to utility of masked actions
        if mask:
            utility[~mask] = -1e8

        # Add molecules to action
        molecules = self._gather_molecules(
            observations, utility, batch_size=action.pop('batch_size'))
        action.update({'molecules': molecules})

        return action

    def policy(self, observations):
        raise NotImplementedError

    def construct_action(self, molecules):
        raise NotImplementedError
    
    def _filter_observations(
        self,
        observations,
        action: Optional[dict] = None
    ):
        # Normalize input
        action_name = dg.utils.normalize_list(action['name'])[0]
        
        # Filter observations
        match action_name:
            case 'design':
                observations = (observations.scored + observations.tested) or observations
            case 'make':
                observations = observations.scored + observations.designed
            case _ as test:
                if 'Noisy' in test:
                    observations = observations.designed
                else:
                    observations = observations.made

        return observations
    
    def _gather_molecules(
        self,
        observations,
        utility: Iterable,
        batch_size: int,
    ):
        batch_size = min([batch_size, len(observations)])
        pending = self.exploration_strategy(utility, size=batch_size)
        molecules = [observations.index[p] for p in pending]
        return molecules
    
    def _expand_action(self, action: dict):
        """
        Transforms a dictionary with any keys containing lists into a list of dictionaries,
        representing all possible combinations of list elements for each key.
        Non-list values are copied directly.

        """
        list_keys = {k: v for k, v in action.items() if isinstance(v, list)}
        constant_keys = {k: v for k, v in action.items() if not isinstance(v, list)}
        
        if not list_keys:
            return [action]

        # Loop Cartesian product of all action keys
        keys, values_lists = zip(*list_keys.items())
        combinations = list(itertools.product(*values_lists))
        
        actions = []
        for combination in combinations:
            combined_dict = dict(zip(keys, combination))
            combined_dict.update(constant_keys)
            actions.append(combined_dict)

        return actions


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
