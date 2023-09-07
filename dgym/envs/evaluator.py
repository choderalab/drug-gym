import dgym as dg
import numpy as np
from dgym.envs.oracle import Oracle
from typing import Optional, Callable


class UtilityFunction:
    """
    Example
    -------

    >>> # Define evaluators for log S and log P
    >>> log_S_evaluator = PropertyEvaluator(
    >>>     ideal=(-2, 0),
    >>>     acceptable=(-4, 0.5)
    >>> )
    >>>
    >>> # Define evaluators for log P
    >>> log_P_evaluator = PropertyEvaluator(
    >>>     ideal=(1, 4),
    >>>     acceptable=(0, 5)
    >>> )
    >>>
    >>> utility = UtilityFunction(
    >>>     evaluators = [log_S_evaluator, log_P_evaluator],
    >>>     strategy = lambda x: np.prod(x)
    >>> )
    >>> 
    >>> assert utility.score([0, 5]) == 0.5458775937413701

    """
    def __init__(self, oracles, evaluators, strategy: Callable):
        
        # Check correspondence of oracles and evaluators
        assert len(oracles) == len(evaluators)
        assert all(isinstance(oracle, Oracle) for oracle in oracles)
        assert all(isinstance(evaluator, Evaluator) for evaluator in evaluators)

        self.oracles = oracles
        self.evaluators = evaluators
        self.strategy = strategy

    def score(self, molecules):
        """Compute the score for molecules based on their properties."""
        if not molecules:
            return None
        
        scores = [
            evaluator.score(oracle(molecules))
            for evaluator, oracle in zip(self.evaluators, self.oracles)
        ]

        return self.strategy(scores)

    def plot(self, deck, **kwargs):
        return dg.plotting.plot(deck, self, **kwargs)

    def __call__(self, values):
        return self.score(values)


class Evaluator:
    
    def __init__(self, ideal, acceptable):
        self.ideal = np.array(ideal)
        self.acceptable = np.array(acceptable)

    def score(self, value):
        is_scalar = np.isscalar(value)
        value = np.asarray(value)
        scores = np.where(
            (self.ideal[0] <= value) & (value <= self.ideal[1]), 
            1, 
            self.score_acceptable(value)
        )
        return scores.item() if is_scalar else scores

    def score_acceptable(self, value):
        raise NotImplementedError


class ClassicEvaluator(Evaluator):
    
    def __init__(self, ideal, acceptable):
        super().__init__(ideal, acceptable)
 
        self._lower_slope = self._slope(self.acceptable[0], 0.5, self.ideal[0], 1)
        self._upper_slope = self._slope(self.ideal[1], 1, self.acceptable[1], 0.5)

    def score_acceptable(self, value):
        value = np.asarray(value)
        res = np.empty_like(value, dtype=float)
        
        # Value less than lower acceptable limit
        mask = value < self.acceptable[0]
        res[mask] = self._logistic(value[mask], 1, self.acceptable[0])
        
        # Value between lower acceptable limit and lower ideal limit
        mask = (self.acceptable[0] < value) & (value < self.ideal[0])
        res[mask] = (value[mask] - self.ideal[0]) * self._lower_slope + 1
        
        # Value between upper ideal limit and upper acceptable limit
        mask = (self.ideal[1] < value) & (value < self.acceptable[1])
        res[mask] = (value[mask] - self.ideal[1]) * self._upper_slope + 1
        
        # Value greater than upper acceptable limit
        mask = value > self.acceptable[1]
        res[mask] = self._logistic(value[mask], -1, self.acceptable[1])
        
        return res

    @staticmethod
    def _slope(x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0

    @staticmethod
    def _logistic(x, scale, midpoint):
        return 1 / (1 + np.exp((midpoint - x) * scale))
