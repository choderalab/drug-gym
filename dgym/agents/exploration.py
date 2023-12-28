import random
from typing import Iterable
from abc import abstractmethod

class ExplorationStrategy:
    
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, utility: Iterable, size: int) -> list:
        raise NotImplementedError
    

class EpsilonGreedy(ExplorationStrategy):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def select_action(self, utility, size):
        
        action = []
        indices = utility.argsort().tolist()
        
        for _ in range(size):
            if random.random() < self.epsilon:
                index = random.randrange(len(indices))
                action.append(indices.pop(index))
            else:
                action.append(indices.pop())
        
        return action
    
    def __call__(self, utility, size):
        return self.select_action(utility, size)