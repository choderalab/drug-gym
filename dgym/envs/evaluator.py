import numpy as np
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
    >>> utility.score([0, 5])
    
    0.5458775937413701


    """
    def __init__(self, evaluators, strategy: Optional[Callable] = None):
        """Initializes with a list of PropertyEvaluator instances."""
        self.evaluators = evaluators
        self._strategy = strategy
    
    def score(self, values):
        """Compute the score for a molecule based on its properties."""
        if not values: return None
        scores = [evaluator.score(val)
                  for evaluator, val
                  in zip(self.evaluators, values)]
        return self.strategy(scores)

    def strategy(self, scores):
        if self._strategy:
            return self._strategy(scores)
        else:
            # Default behavior or raise an exception
            raise NotImplementedError("A strategy function must be provided or set.")

    def __call__(self, values):
        return self.score(values)


class PropertyEvaluator:
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

class GaussianPropertyEvaluator(PropertyEvaluator):
    def __init__(self, ideal, acceptable):
        
        super().__init__(ideal, acceptable)
        self.mu_acceptable = np.mean(acceptable)
        self.sigma_acceptable = (acceptable[1] - acceptable[0]) / 2
    
    def score_acceptable(self, value):        
        return 0.9 * self._gaussian(value, self.mu_acceptable, self.sigma_acceptable)

    def _gaussian(x, mu, sigma, A=1):
        """Computes the Gaussian function for a given x."""
        sigma = sigma if sigma != 0 else 1e-6
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
