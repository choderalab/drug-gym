import rdkit
import torch
from typing import Optional, List, Any
from dgym.reaction import Reaction
from rdkit.Chem.rdChemReactions import ChemicalReaction

class Repertoire:
    def __init__(self, reactions: Optional[List] = None) -> None:
        if reactions is None:
            reactions = []
        assert isinstance(reactions, List)
        assert all(isinstance(reaction, Reaction) for reaction in reactions)
        self.reactions = reactions

    def __len__(self):
        """Return the number of molecules in the collection."""
        if self.reactions is None:
            return 0
        return len(self.reactions)

    def __getitem__(self, key: Any):
        """Get item from the collection.
        Parameters
        ----------
        key : Any
        Notes
        -----
        * If the key is integer, return the single molecule indexed.
        * If the key is a string, return a collection of all molecules with
            this SMILES.
        * If the key is a molecule, extract the SMILES string and index by
            its SMILES.
        * If the key is a tensor, flatten it to treat it as a list.
        * If the key is a list, return a collection with molecules indexed by
            the elements in the list.
        * If the key is a slice, slice the range and treat at as a list.
        """
        if self.reactions is None:
            raise RuntimeError("Empty Repertoire.")
        if isinstance(key, int):
            return self.reactions[key]
        # elif isinstance(key, str):
        #     return self.__class__(molecules=[self.lookup[key]])
        # elif isinstance(key, Molecule):
        #     return self.lookup[key.smiles]
        elif isinstance(key, torch.Tensor):
            key = key.detach().flatten().cpu().numpy().tolist()
        elif isinstance(key, list):
            return self.__class__(
                reactions=[self.reactions[_idx] for _idx in key]
            )
        elif isinstance(key, slice):
            return self.__class__(reactions=self.reactions[key])
        else:
            raise RuntimeError("The slice is not recognized.")

    def __repr__(self):
        return "%s with %s reactions" % (self.__class__.__name__, len(self))


def from_json(
    path: str,
    smarts_col: str,
    classes_col: Optional[str] = None
):
    """
    Load reactions.
    From SmilesClickChem: https://zenodo.org/record/4100676
    
    """
    import pandas as pd

    # load from JSON
    reactions_df = pd.read_json(path).T.reset_index(drop=True)
        
    def _make_reaction(row):
        smirks = row[smarts_col]
        metadata = {'name': row['reaction_name']}
        r = Reaction(smirks, metadata=metadata)
        r = r.annotate_reactants(row[classes_col])
        return r

    repertoire = reactions_df.apply(_make_reaction, axis=1).tolist()
    return Repertoire(repertoire)
