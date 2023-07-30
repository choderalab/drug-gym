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
        reactants: Optional[Iterable] = None,
        annotations: Optional[dict] = None,
        id_attr: Optional[str] = 'smiles',
        # featurizer: Optional[Callable] = functools.partial(
        #     smiles_to_bigraph,
        #     node_featurizer=CanonicalAtomFeaturizer(atom_data_field="h"),
        # ),
    ) -> None:
        
        if isinstance(mol, str):
            mol = rdkit.Chem.MolFromSmiles(mol)

        self.mol = mol
        self.reactants = reactants
        self.smiles = rdkit.Chem.MolToSmiles(self.mol)
        self._id_attr = id_attr
        
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

    def poise(self, idx):

        assert hasattr(self, 'reactants')
        
        def _move_idx_to_first(lst, idx):
            lst.insert(0, lst.pop(idx))
            return lst

        # if only one reactant, do nothing
        if len(self.reactants) == 1:
            return self

        # do not mutate in-place
        from copy import deepcopy
        temp = deepcopy(self)
        
        # reorder according to poised index
        temp.reactants = _move_idx_to_first(temp.reactants, idx)

        return temp

    # def __repr__(self) -> str:
    #     return self.smiles

    def __getitem__(self, idx):
        if not self.metadata:
            raise RuntimeError("No data associated with Molecule.")
        elif isinstance(idx, str):
            return self.metadata[idx]
        elif idx is None and len(self.metadata) == 1:
            return list(self.metadata.values())[0]
        else:
            raise NotImplementedError

    def featurize(self) -> None:
        """Featurize the SMILES string to get the graph.
        Returns
        -------
        dgl.DGLGraph : The resulting graph.
        """
        # if there is already a graph, do nothing
        if not self.is_featurized():
            # featurize
            self.g = self.featurizer(self.smiles)

        return self

    def is_featurized(self) -> bool:
        """Returns whether this molecule is attached with a graph. """
        return self.g is not None

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
        >>> molecule = Molecule("C", metadata={"name": "john"})
        Type mismatch:
        >>> molecule == "john"
        False
        Graph mismatch:
        >>> molecule == Molecule("CC", metadata={"name": "john"})
        False
        Metadata mismatch:
        >>> molecule == Molecule("C", metadata={"name": "jane"})
        False
        Both graph and metadata match:
        >>> molecule == Molecule("C", metadata={"name": "john"})
        True
        """
        # if not a molecule, fuggedaboutit
        if not isinstance(other, type(self)):
            return False

        # NOTE(yuanqing-wang):
        # Equality is not well-defined for DGL graph
        # Use networx isomorphism instead.
        import networkx as nx
        return (
            nx.is_isomorphic(self.g.to_networkx(), other.g.to_networkx())
            and self.metadata == other.metadata
        )

    def erase_annotations(self) -> Any:
        """Erase the metadata. """
        self.metadata = None
        return self

    def update_annotations(self, other_annotations: Optional[dict] = None) -> Any:
        """Update annotations. """

        self.annotations.update(self.mol.GetPropsAsDict())
        
        if other_annotations:
            self.annotations.update(other_annotations)
        
            # synchronize with rdkit mol
            for key, value in self.annotations.items():
                self.mol.SetProp(str(key), value)

        return self

    def update_cache(self):
        self.mol.UpdatePropertyCache()
        return self