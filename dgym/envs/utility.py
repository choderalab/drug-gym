import dgym as dg
import numpy as np
from dgym.molecule import Molecule
from dgym.envs.oracle import Oracle
from typing import Optional, Callable, Iterable


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

    def __call__(self, input):
        
        # Normalize input
        return_list = isinstance(input, Iterable)
        input = input if return_list else np.array([input])
        
        # Score molecules
        if isinstance(input[0], Molecule):
            input = self.oracle(input)

        # Normalize scores
        scores = self.score(input)
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
        
        return res
