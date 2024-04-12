import json
import numpy as np
from tqdm import tqdm
from typing import Optional
from dgym.utils import serialize_with_class_names

class Experiment:
    
    def __init__(self, drug_agent, drug_env):
        
        self.drug_agent = drug_agent
        self.drug_env = drug_env
    
    def run(
        self,
        num_trials=1,
        progress=True,
        out: Optional[str] = None,
        **kwargs
    ):

        results = []
        for trial in tqdm(range(num_trials)):

            observations, info = self.drug_env.reset()

            if progress:
                pbar = tqdm(total = self.drug_env.budget)
            
            while True:
                
                # Perform step
                action = self.drug_agent.act(observations)
                observations, _, terminated, truncated, _ = self.drug_env.step(action)
                
                # Parse result
                result = self.get_result(trial, out=out, **kwargs)
                
                if progress:
                    pbar.n = len(self.drug_env.library.tested)
                    pbar.update()

                if terminated or truncated:
                    break

            if terminated:
                result.update({'outcome': 1})
            elif truncated:
                result.update({'outcome': 0})

            results.append(result)

        return results

    def get_result(
        self,
        trial: int,
        out: Optional[str] = None,
        **kwargs
    ):
        result = {
            'trial': trial,
            'cost': len(self.drug_env.library),
            'time_elapsed': self.drug_env.time_elapsed,
            'annotations': self.drug_env.library.annotations.to_dict(),
            **vars(self.drug_agent),
            **kwargs
        }
        
        if out:
            result_serialized = serialize_with_class_names(result)
            json.dump(result_serialized, open(out, 'w'))
            
        return result