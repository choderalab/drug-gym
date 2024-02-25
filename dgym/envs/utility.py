import dgym as dg
import numpy as np
from dgym.molecule import Molecule
from dgym.envs.oracle import Oracle
from typing import Optional, Callable, Iterable


class UtilityFunction:
    
    def __init__(self, oracle, ideal, acceptable):

        self.oracle = oracle
        self.ideal = np.array(ideal)
        self.acceptable = np.array(acceptable)

    def __call__(self, input):
        
        if isinstance(input[0], Molecule):
            input = self.oracle(input)
        
        return self.score(input)

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


class ClassicUtilityFunction(UtilityFunction):
    
    def __init__(self, oracle, ideal, acceptable):
        super().__init__(oracle, ideal, acceptable)
 
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
