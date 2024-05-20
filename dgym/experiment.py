import ast
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional
from dgym.utils import serialize_with_class_names, print_memory_usage
from dgym.envs import DrugEnv
from dgym.agents import DrugAgent
from dgym.molecule import Molecule
from dgym.collection import MoleculeCollection

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
            print(observations)

            if progress:
                pbar = tqdm(total = self.drug_env.budget)
            
            while True:
                
                # Perform step
                action = self.drug_agent.act(observations)
                print('Created action', flush=True)
                print(action, flush=True)
                observations, _, terminated, truncated, _ = self.drug_env.step(action)
                
                # Parse result
                result = self.dump(trial, out=out, **kwargs)
                print(self.drug_env.get_reward(), flush=True)
                
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

    def dump(
        self,
        trial: int,
        out: Optional[str] = None,
        **kwargs
    ):
        annotations = self.drug_env.library.annotations.reindex(
            columns=[
                'SMILES',
                'Synthetic Route',
                'Inspiration',
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
            with open(out, 'w') as f:
                json.dump(result_serialized, f)
            
        return result
    
    def load(self, result):
        """
        Load a result JSON from `get_result` and return MoleculeCollection for loading DrugEnv.
        
        Usage
        -----
        ```
        result = experiment.dump(trial=1)
        experiment = experiment.load(result)
        ```
        """
        if result:
            annotations = pd.DataFrame(result['annotations'])
            molecules = []
            for _, annotation in annotations.iterrows():

                # Parse data structure
                annotation = annotation.to_dict()
                route = annotation.pop('Synthetic Route')
                try:
                    route = ast.literal_eval(route)
                except:
                    pass
                route['annotations'] = annotation

                # Load molecule
                molecule = Molecule.load(route)

                # Append to library
                molecules.append(molecule)

            # Load collection
            self.drug_env.library = MoleculeCollection(molecules)
        
        return self