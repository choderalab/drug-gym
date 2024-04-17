import json
import numpy as np
from tqdm.auto import tqdm
from typing import Optional
from dgym.utils import serialize_with_class_names
from dgym.envs import DrugEnv
from dgym.agents import DrugAgent

class Experiment:
    
    def __init__(
        self,
        drug_agent: DrugAgent,
        drug_env: DrugEnv
    ):
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

            self.drug_agent.reset()
            observations, info = self.drug_env.reset()

            if progress:
                pbar = tqdm(total = self.drug_env.budget)
            
            while True:
                
                # Perform step
                action = self.drug_agent.act(observations)
                observations, _, terminated, truncated, _ = self.drug_env.step(action)
                
                # Parse result
                result = self.get_result(trial, out=out, **kwargs)
                print(self.drug_env.get_reward())
                
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
        annotations = self.drug_env.library.annotations.reindex(
            columns=[
                'SMILES',
                'Reactants',
                'Step Designed',
                'Step Scored',
                'Step Made',
                'Step Tested',
                'Current Status',
                *self.drug_env.assays
            ]
        ).to_dict()

        result = {
            'trial': trial,
            'cost': len(self.drug_env.library.tested),
            'time_elapsed': self.drug_env.time_elapsed,
            'annotations': annotations,
            **vars(self.drug_agent),
            **kwargs
        }
        
        if out:
            result_serialized = serialize_with_class_names(result)
            json.dump(result_serialized, open(out, 'w'))
            
        return result