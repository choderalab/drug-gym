import numpy as np
from tqdm import tqdm

class Experiment:
    
    def __init__(self, drug_agent, drug_env):
        
        self.drug_agent = drug_agent
        self.drug_env = drug_env
    
    def run(self, num_trials=1, progress=False, **kwargs):

        results = []
        for trial in tqdm(range(num_trials)):

            observations, info = self.drug_env.reset()

            if progress:
                pbar = tqdm(total = self.drug_env.budget)
            
            while True:
                action = self.drug_agent.act(observations)
                observations, _, terminated, truncated, _ = self.drug_env.step(action)
                
                try:
                    print(np.nanmax(observations.annotations['ADAM17 affinity']))
                except:
                    pass

                if progress:
                    pbar.n = len(self.drug_env.library)
                    pbar.update()
                
                if terminated or truncated:
                    break

            result = {
                'trial': trial,
                **vars(self.drug_agent),
                **kwargs
            }

            if terminated:
                cost = len(self.drug_env.library)
                result.update({'cost': cost, 'outcome': 1})

            if truncated:
                result.update({'outcome': 0})

            results.append(result)

        return results