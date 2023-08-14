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
        assert all(isinstance(evaluators, Evaluator) for evaluator in evaluators)

        self.oracles = oracles
        self.evaluators = evaluators
        self.strategy = strategy

    def score(self, molecules):
        """Compute the score for a molecule based on its properties."""
        if not values: return None

        scores = [evaluator.score(val)
                  for evaluator, val
                  in zip(self.evaluators, values)]
        return self.strategy(scores)

    def __call__(self, values):
        return self.score(values)


class Evaluator:
    def __init__(self, ideal, acceptable):
        self.ideal = ideal
        self.acceptable = acceptable
    
    def score(self, value):
        if self.ideal[0] <= value <= self.ideal[1]:
            return 1
        else:
            return self.score_acceptable(value)

    def score_acceptable(self, value):
        raise NotImplementedError

class ClassicEvaluator(Evaluator):
    def __init__(self, ideal, acceptable):
        
        super().__init__(ideal, acceptable)
        self._lower_slope = self._slope(self.acceptable[0], 0.5, self.ideal[0], 1)
        self._upper_slope = self._slope(self.ideal[1], 1, self.acceptable[1], 0.5)
    
    def score_acceptable(self, value):
        
        if value < self.acceptable[0]:
            return self._logistic(value, 1, self.acceptable[0]) # positive slope
        
        elif value > self.acceptable[0] and value < self.ideal[0]:
            return (value - self.ideal[0]) * self._lower_slope + 1
        
        elif value < self.acceptable[1] and value > self.ideal[1]:
            return (value - self.ideal[1]) * self._upper_slope + 1
        
        if value > self.acceptable[1]:
            return self._logistic(value, -1, self.acceptable[1]) # negative slope

    @staticmethod
    def _slope(x1, y1, x2, y2):
        if(x2 - x1 != 0):
          return (float)(y2-y1)/(x2-x1)

    @staticmethod
    def _logistic(x, scale, midpoint):
        """ Slope of logistic function is scale / 4 """
        return 1 / (1 + np.exp((midpoint - x) * scale))
