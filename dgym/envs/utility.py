import numbers
import dgym as dg
import numpy as np
import itertools
from dgym.molecule import Molecule
from dgym.envs.oracle import Oracle
from typing import Optional, Callable, Iterable
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import pandas as pd

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


class Policy:
    
    def __init__(
        self,
        utility_functions: Iterable[UtilityFunction],
        weights: Optional[Iterable[float]] = None,
    ):
        self.utility_functions = utility_functions
        self.weights = weights
    
    @property
    def oracle_names(self):
        return [u_fn.oracle.name for u_fn in self.utility_functions]

    def __call__(
        self,
        input: Iterable,
        method: str = 'hybrid',
        use_precomputed: bool = False,
        rescale: bool = True,
        **kwargs
    ):
        # Normalize inputs
        return_list = isinstance(input, Iterable) \
            and (isinstance(input[0], Iterable) or isinstance(input[0], Molecule))
        
        if use_precomputed:
            input = self._get_precomputed(input, rescale=rescale)         

        # Score molecules
        utility = self.score(input, **kwargs)

        # Compose across objectives
        composite_utility = self.compose(utility, method=method)
        
        return composite_utility.tolist() if return_list else composite_utility.item()
    
    def _get_precomputed(
        self,
        input: Iterable,
        rescale: bool = True
    ):
        """
        Gracefully grabs precomputed data. Merges actual and surrogate data, preferring actual.
        Uses ordinary least squares to correct selection bias in surrogate model scores.
        """
        # Normalize input
        annotations = input.annotations
        if not isinstance(annotations, pd.DataFrame):
            annotations = pd.DataFrame([annotations])

        # Get actual data
        actuals = annotations.reindex(columns=self.oracle_names)
        
        # Get surrogate data
        surrogates = actuals.add_prefix('Noisy ').columns
        surrogates = annotations.reindex(columns=surrogates)
        surrogates.columns = surrogates.columns.str.removeprefix('Noisy ')
                
        # Correct selection bias with OLS
        if rescale:
            surrogates = self._rescale_surrogates(surrogates, actuals)

        # Merge actuals and surrogates
        annotations = actuals.combine_first(surrogates).values
        
        return annotations
    
    def _rescale_surrogates(self, surrogates, actuals):
        """
        Rescales surrogates to reduce selection bias.
        """
        if len(actuals.dropna()) <= 30:
            return surrogates

        # Create filter for complete measurements
        check_complete = lambda df: ~df.isna().any(axis=1)
        is_complete = check_complete(actuals) & check_complete(surrogates)

        # Rescale each column        
        for column in actuals:
            
            actuals_subset = actuals[column][is_complete]

            surrogates_column = sm.add_constant(surrogates[column])
            surrogates_subset = surrogates_column[is_complete]

            regressor = OLS(actuals_subset, surrogates_subset).fit()
            surrogates[column] = regressor.predict(surrogates_column)

        return surrogates
    
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
                    composite_utility = np.array([1])
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
    