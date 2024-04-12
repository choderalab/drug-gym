import numbers
import dgym as dg
import numpy as np
import itertools
from dgym.molecule import Molecule
from dgym.envs.oracle import Oracle
from typing import Optional, Callable, Iterable
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class UtilityFunction:

    def __init__(
        self,
        oracle: Optional[Oracle] = None,
        ideal: Iterable = [],
        acceptable: Iterable = []
    ):
        self.oracle = oracle
        self.ideal = np.array(ideal)
        self.acceptable = np.array(acceptable)

    def __call__(self, input, **kwargs):

        # Normalize input
        return_list = isinstance(input, Iterable)
        input = input if return_list else np.array([input])
        
        # Score molecules
        if isinstance(input[0], Molecule):
            input = self.oracle(input, **kwargs)

        # Normalize scores
        scores = self.score(input)
        scores[np.isnan(input)] = -1e3

        return scores if return_list else scores.item()

    def score(self, value):
        raise NotImplementedError

class ClassicUtilityFunction(UtilityFunction):
    
    def __init__(
        self,
        oracle: Optional[Oracle] = None,
        ideal: Iterable = [],
        acceptable: Iterable = []
    ):
        super().__init__(oracle, ideal, acceptable)

    def score(self, value):
        value = np.asarray(value)
        res = np.empty_like(value, dtype=float)
        
        # Value less than lower acceptable limit (quadratic penalty)
        mask = value < self.acceptable[0]
        res[mask] = (abs(value[mask] - self.acceptable[0]) + 1)**2
        
        # Value between lower acceptable limit and lower ideal limit (linear interpolation)
        mask = (self.acceptable[0] <= value) & (value < self.ideal[0])
        res[mask] = np.interp(value[mask], [self.acceptable[0], self.ideal[0]], [1, 0])
        
        # Value inside ideal range
        mask = (self.ideal[0] <= value) & (value <= self.ideal[1])
        res[mask] = 0
        
        # Value between upper ideal limit and upper acceptable limit (linear interpolation)
        mask = (self.ideal[1] < value) & (value <= self.acceptable[1])
        res[mask] = np.interp(value[mask], [self.ideal[1], self.acceptable[1]], [0, 1])
        
        # Value greater than upper acceptable limit (quadratic penalty)
        mask = value > self.acceptable[1]
        res[mask] = (abs(value[mask] - self.acceptable[1]) + 1)**2
        
        return 1 - res


class MultipleUtilityFunction:
    
    def __init__(
        self,
        utility_functions: Iterable[UtilityFunction],
        weights: Optional[Iterable[float]] = None,
    ):
        self.utility_functions = utility_functions
        self.weights = weights
    
    def __call__(
        self,
        input,
        method: str = 'hybrid',
        **kwargs
    ):
        # Score molecules
        utility = self.score(input, **kwargs)

        # Compose across objectives
        composite_utility = self.compose(utility, method=method)
        
        return composite_utility.tolist()
    
    def score(
        self,
        input,
        **kwargs
    ):
        
        # Initialize utility array
        input_size = len(input) if not isinstance(input[0], numbers.Number) else 1
        utility = np.empty((input_size, len(self.utility_functions)))

        # Normalize input
        if isinstance(input[0], Iterable):
            input = np.array(input).T
        elif isinstance(input[0], Molecule):
            input = itertools.repeat(input)

        # Compute utility
        for idx, (utility_function, input_) in enumerate(zip(self.utility_functions, input)):
            utility[:,idx] = utility_function(input_, **kwargs)            
        
        return utility

    def compose(
        self,
        utility: Iterable,
        method: str = 'hybrid'
    ):
        match method:
            case 'hybrid':
                if len(utility) == 1:
                    composite_utility = 1
                else:
                    nds_ranks = self._non_dominated_sort(utility)
                    averages = self._weighted_average(utility)
                    hybrid_sort = np.lexsort([-averages, nds_ranks])
                    hybrid_ranks = hybrid_sort.argsort()
                    composite_utility = 1 - (hybrid_ranks / (len(hybrid_ranks) - 1))
            case 'average':
                composite_utility = self._weighted_average(utility)
            case 'max':
                composite_utility = np.nanmax(utility, axis=1)
            case 'min':
                composite_utility = np.nanmin(utility, axis=1)
        
        return composite_utility

    def _non_dominated_sort(self, utility):
        _, nds_ranks = NonDominatedSorting().do(
            -utility,
            return_rank=True,
            only_non_dominated_front=False
        )
        return nds_ranks
    
    def _weighted_average(self, utility):
        return np.average(utility, axis=1, weights=self.weights)