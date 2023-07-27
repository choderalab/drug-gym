import dgl
import dgllife
import numpy as np
from rdkit import Chem
from typing import Optional
from collections.abc import Callable


class DrugAgent(object):

    def __init__(
        self,
        collection: Optional[dg.Collection] = dg.Collection()
    ) -> None:

        self.collection = collection

    def sample_action(self, obs: list, num_orders: int):
        """
        Returns `orders`: tuples of `(service, list(Molecule))`.

        """
        raise NotImplementedError

class OneStepSyntheticDrugAgent(DrugAgent):
    
    def __init__(
        self,
        repertoire: list,
        building_blocks: dict,
        collection: Optional[dg.Collection] = dg.Collection()
    ) -> None:

        super().__init__(self, collection)
        self.repertoire = repertoire
        self.building_blocks = building_blocks

    def sample_action(
        self,
        compounds: Optional[dg.Collection] = None,
        properties=None,
        num_analogs: int,
        num_orders: int,
        epsilon_orders: float,
        sortby: Union[str, dict] = {'fingerprint': 0.5, 'random': 0.5},
        fps: Union[chemfp.arena.FingerprintArena, None] = None,
    ):
        """
        Enumerates and featurizes candidates.
        Then act according to policy to place orders.

        Returns `orders`: tuples of `(service, list(Molecule))`.
        
        """
        if not compounds:
            compounds = self.collection

        analogs = []
        for compound in compounds:
            analogs.extend(
                dg.synthesis.enumerate_analogs(
                    compound,
                    self.repertoire,
                    self.building_blocks,
                    num_analogs=num_analogs,
                    sortby=sortby,
                    fps=fps
                )
            )
        feats = self._featurize(analogs, properties=properties)
        orders = self._act(analogs, feats, k=k)
        return orders

    
    
    
    def _featurize(
        self,
        molecules,
        featurizer: Optional[Callable],
        properties=['GCN_canonical_Lipophilicity']
    ):
        """
        Featurize a set of molecules.

        """
        # import pdb; pdb.set_trace()
        feats = get_properties(
            collection.Collection(molecules),
            properties=properties
        )
        return feats


    def _act(
        self,
        molecules,
        feats,
        k=1,
        policy='greedy',
        assay='GCN_canonical_Lipophilicity'
    ):
        if policy == 'greedy':
            indices = np.argsort(feats)[-k:]
            chosen_molecules = collection.Collection(
                [molecules[i] for i in indices]
            )
        orders = tuple([assay, chosen_molecules])
        return orders
