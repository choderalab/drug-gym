"""Models information associated with a molecule."""
# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union, Any, Optional, Iterable, Mapping, Callable, Sequence
import functools
import dgl
import rdkit
import copy
import torch
from rdkit.Chem import Mol
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Molecule:
    """ Models information associated with a molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES of the molecule.
    g : dgl.DGLGraph or None, default=None
        The DGL graph of the molecule.
    metadata : Any
        Metadata associated with the molecule.
    featurizer : callable, default=CanonicalAtomFeaturizer(
    atom_data_field='feat')
        The function which maps the SMILES string to a DGL graph.
    
    Methods
    -------
    featurize()
        Convert the SMILES string to a graph if there isn't one.
    Notes
    -----
    * The current CanonicalAtomFeaturizer has implicit Hs.
    
    Examples
    --------
    >>> molecule = Molecule("C")
    >>> molecule.g.number_of_nodes()
    1
    """
    def __init__(
        self,
        mol: Optional[rdkit.Chem.Mol] = None,
        reactants: Optional[Iterable] = [],
        id_attr: Optional[str] = 'smiles',
        reaction: Optional[str] = None,
        inspiration: Optional[Iterable] = None,
        annotations: Optional[dict] = None,
    ) -> None:
        
        if isinstance(mol, str):
            mol = rdkit.Chem.MolFromSmiles(mol)

        reactants = list(reactants)
        for idx, reactant in enumerate(reactants):
            if isinstance(reactant, Mol):
                reactants[idx] = Molecule(reactant)

        self.mol = mol
        self.reactants = reactants
        self.reaction = reaction
        self.inspiration = inspiration
        self._id_attr = id_attr
        self.smiles = rdkit.Chem.CanonSmiles(
            rdkit.Chem.MolToSmiles(self.mol)
        )
        
        if annotations is None:
            annotations = {}
        self.annotations = annotations
        self.update_annotations()


    @property
    def id(self):
        return getattr(self, self._id_attr, None)

    def _repr_html_(self):
        return self.mol._repr_html_()

    def _repr_png_(self):
        return self.mol._repr_png_()

    def has_substruct_match(self, template):
        return self.mol.HasSubstructMatch(template)

    def annotate_reactants(self, classes):

        assert len(self.reactants) == len(classes)
        for i, _ in enumerate(self.reactants):
            self.reactants[i].SetProp('class', classes[i])

        return self

    def __getitem__(self, idx):
        if not self.annotations:
            raise RuntimeError("No annotations associated with Molecule.")
        elif isinstance(idx, str):
            return self.annotations[idx]
        elif idx is None and len(self.annotations) == 1:
            return list(self.annotations.values())[0]
        else:
            raise NotImplementedError

    def __eq__(self, other: Any):
        """Determine if two AssayedMolecule objects are equal.
        Parameters
        ----------
        other : Any
            The other object
        Returns
        -------
        bool
            If the two objects are identical.
        Examples
        --------
        TODO
        """
        # if not a molecule, fuggedaboutit
        if not isinstance(other, type(self)):
            return False
        return self.smiles == other.smiles

    def __hash__(self):
        return hash(self.smiles)

    def erase_annotations(self) -> Any:
        """Erase the metadata. """
        self.metadata = None
        return self

    def update_annotations(self, other_annotations: Optional[dict] = None) -> Any:
        """Update annotations. """

        self.annotations.update(self.mol.GetPropsAsDict())

        if 'smiles' not in self.annotations:
            self.annotations.update({'smiles': self.smiles})
        
        if other_annotations:
            self.annotations.update(other_annotations)
        
            # synchronize with rdkit mol
            for key, value in self.annotations.items():
                self.mol.SetProp(str(key), str(value))

        return self

    def update_cache(self):
        try:
            self.mol.UpdatePropertyCache()
        except:
            # sometimes throws AtomValenceException
            pass
        return self
