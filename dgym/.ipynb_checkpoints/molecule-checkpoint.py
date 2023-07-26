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
class Molecule(object):
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
        # smiles: str,
        # g: Optional[dgl.DGLGraph] = None,
        metadata: Optional[dict] = {},
        # featurizer: Optional[Callable] = functools.partial(
        #     smiles_to_bigraph,
        #     node_featurizer=CanonicalAtomFeaturizer(atom_data_field="h"),
        # ),
    ) -> None:
        
        if isinstance(mol, str):
            mol = rdkit.Chem.MolFromSmiles(mol)

        self.mol = mol
        self.smiles = rdkit.Chem.MolToSmiles(self.mol)
        self.reactants = reactants
        # self.smiles = smiles
        # self.g = g
        self.metadata = metadata
        
        # Set the properties of the molecule using the dictionary
        for key, value in metadata.items():
            self.mol.SetProp(key, value)

        # self.featurizer = featurizer

        # featurize the first thing after init
        # self.featurize()

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

    def erase_annotation(self) -> Any:
        """Erase the metadata. """
        self.metadata = None
        return self