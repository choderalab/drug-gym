"""Collections and their serving."""
# =============================================================================
# IMPORTS
# =============================================================================
import copy
import random
import dgym as dg
import torch
import random
import itertools
import numpy as np
import pandas as pd
from rdkit import Chem
from functools import partial
from dgym.molecule import Molecule
from dgym.reaction import LazyReaction
from typing import Union, Iterable, Optional, List, Any, Callable

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Collection(torch.utils.data.Dataset):
    """A collection of Molecules with functionalities to be compatible with
    training and optimization.
    
    Parameters
    ----------
    molecules : List[drug_gym.Molecule]
        A list of Molecules.
    
    Methods
    -------
    featurize(molecules)
        Featurize all molecules in the collection.
    view()
        Generate a torch.utils.data.DataLoader from this Collection.
    
    """
    def __init__(
        self,
        items: Optional[Iterable] = [],
        index: Optional[Iterable] = []
    ) -> None:
        super(Collection, self).__init__()
        assert isinstance(items, Iterable)
        
        self._items = items
        self._lookup = None
        self._index = list(range(len(items))) \
            if not index else index

    def __repr__(self):
        
        if self._items:
            repr_ = "{collection} with {size} {items}s".format(
                collection = self.__class__.__name__,
                size = len(self),
                items = self._items[0].__class__.__name__
            )
        
        else:
            repr_ = f"Empty {self.__class__.__name__}"
        
        return repr_

    def _construct_lookup(self):
        """Construct lookup table for molecules."""
        self._lookup = {item.name: item for item in self._items}

    @property
    def index(self):
        return self._index.copy()
    
    def reset_index(self):
        self._index = list(range(len(self._items)))
        return self

    @property
    def annotations(self):
        self.update_annotations()
        return pd.DataFrame(
            [item.annotations for item in self._items],
            index=self.index
        )

    @property
    def annotated(self):
        return [i for i in self._items if i.annotations]
    
    @property
    def lookup(self):
        """Returns the mapping between the SMILES and the molecule. """
        if self._lookup is None:
            self._construct_lookup()
        return self._lookup

    def __contains__(self, item):
        """Check if a molecule is in the collection.
        Parameters
        ----------
        molecule : drug_gym.Molecule
        Examples
        --------
        >>> molecule = Molecule("CC")
        >>> collection = Collection([molecule])
        >>> Molecule("CC") in collection
        True
        >>> Molecule("C") in collection
        False
        """
        return (item is not None) and (item.name in self.lookup)

    def filter(self, by: Callable):
        filtered = [
            [it, ind]
            for (it, ind) in zip(self._items, self.index)
            if by(it)
        ]
        items, index = zip(*filtered) if filtered else ([], [])
        return self.__class__(list(items), index=list(index))

    def apply(self, function):
        """Apply a function to all molecules in the collection.
        Parameters
        ----------
        function : Callable
            The function to be applied to all molecules in this collection
            in place.
        Examples
        --------
        >>> molecule = Molecule("CC")
        >>> collection = Collection([molecule])
        >>> from ..molecule import Molecule
        >>> fn = lambda molecule: Molecule(
        ...     smiles=molecule.smiles, metadata={"name": "john"},
        ... )
        >>> collection = collection.apply(fn)
        >>> collection[0]["name"]
        'john'
        """
        self._items = [function(item) for item in self._items]
        return self

    def __eq__(self, other):
        """Determine if two objects are identical."""
        if not isinstance(other, self.__class__):
            return False
        return self._items == other._items

    def __len__(self):
        """Return the number of molecules in the collection."""
        if self._items is None:
            return 0
        return len(self._items)

    def __getitem__(self, key: Any):
        """Get item from the collection.
        
        Parameters
        ----------
        key : Any
        Notes
        -----
        * If the key is integer, return the single molecule indexed.
        * If the key is a string, the items with this identifier.
        * If the key is a molecule, extract the SMILES string and index by
            its SMILES.
        * If the key is a tensor, flatten it to treat it as a list.
        * If the key is a list, return a collection with molecules indexed by
            the elements in the list.
        * If the key is a slice, slice the range and treat at as a list.
        """
        if not self._items:
            raise RuntimeError("Empty Collection.")
        if isinstance(key, int):
            return self._items[key]
        elif isinstance(key, str):
            return self.lookup[key]
        elif isinstance(key, type(self._items[0])):
            return self.lookup[key.name]
        elif isinstance(key, torch.Tensor):
            key = key.detach().flatten().cpu().numpy().tolist()
        elif isinstance(key, slice):
            return self.__class__(self._items[key], index=self.index[key])
        elif isinstance(key, Iterable):
            if len(key) > 0 and isinstance(key[0], (bool, np.bool_)):
                key = np.where(key)[0]
            return self.__class__(
                [self._items[k] for k in key], index=[self.index[k] for k in key])
        else:
            raise RuntimeError("The slice is not recognized.")

    def __setitem__(self, key: Any, value: Any):
        """Set annotation for the collection."""
        for item in self._items:
            item[key] = value

    def shuffle(self, seed=None):
        """ Shuffle the collection and return it. """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._items)
        return self

    def split(self, partition):
        """Split the collection according to some partition.
        Parameters
        ----------
        partition : Sequence[Optional[int, float]]
            Splitting partition.
        Returns
        -------
        List[Collection]
            List of collections split according to the partition.
        Examples
        --------
        >>> collection = Collection([Molecule("CC"), Molecule("C")])
        >>> collection0, collection1 = collection.split([1, 1])
        >>> collection0[0].smiles
        'CC'
        """
        n_data = len(self)
        partition = [int(n_data * x / sum(partition)) for x in partition]
        parts = []
        idx = 0
        for p_size in partition:
            parts.append(self[idx : idx + p_size])
            idx += p_size
        return parts

    def __add__(self, other):
        """Combine two collections and return a new one.
        Parameters
        ----------
        molecules : Union[List[Molecule], Collection]
            Molecules to be added to the collection.
        Returns
        -------
        >>> collection0 = Collection([Molecule("C")])
        >>> collection1 = Collection([Molecule("CC")])
        >>> collection = collection0 + collection1
        >>> len(collection)
        2
        """
        if isinstance(other, Collection):
            return self.__class__(
                self._items + other._items, index=self.index + other.index)
        elif isinstance(other, Iterable):
            start = max(self.index) + 1 if self.index else 0
            other_index = list(range(start, start + len(other)))
            return self.__class__(
                self._items + list(other), index=self.index + other_index)
        else:
            raise RuntimeError("Addition only supports list and Collection.")

    def __sub__(self, other):
        """ Subtract a list of molecules from a collection and return a new one.
        Parameters
        ----------
        molecules : Union[list[Molecule], Collection]
            Molecules to be subtracted from the collection.
        Returns
        -------
        Collection
            The resulting collection.
        Examples
        --------
        >>> collection = Collection([Molecule("CC"), Molecule("C")])
        >>> collection -= [Molecule("C")]
        >>> len(collection)
        1
        """
        if isinstance(other, list):
            other = self.__class__(other)

        return self.__class__(
            [
                item
                for item in self._items
                if item.name not in other.lookup
            ]
        )

    def __iter__(self):
        """Alias of iter for molecules. """
        return iter(self._items)

    def append(self, item):
        """Append a molecule to the collection.
        Alias of append for molecules.
        Note
        ----
        * This append in-place.
        Parameters
        ----------
        molecule : molecule
            The data molecule to be appended.
        """
        self._items.append(item)
        return self

    def batch(
        self,
        batch_size: int,
    ):
        """
        Generates batches from the collection based on the input specification.

        Parameters:
        - batch_size (int): Specifies the batch.

        Returns:
        - Iterator[Collection]: An iterator over Collections, each being a batch of the original collection.

        Raises:
        - ValueError: If spec is a list and does not sum to approximately 1.0 when it is meant to specify ratios.
        """
        def _batched(iterable, n):
            if n < 1:
                raise ValueError('n must be at least one')
            it = iter(iterable)
            while batch := tuple(itertools.islice(it, n)):
                yield batch

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')

        batched_items = _batched(self._items, batch_size)
        batched_indices = _batched(self.index, batch_size)
        
        for items, indices in zip(batched_items, batched_indices):
            yield self.__class__(list(items), index=list(indices))

    def unique(self):
        return self.__class__(list(set(self._items)))

    def copy(self, unique=False):
        """ Return a copy of self. """
        return self.__class__(copy.deepcopy(self._items))

    def view(
        self,
        collate_fn: Optional[Callable]=None,
        by: Union[Iterable, str] = ['g', 'y'],
        *args,
        **kwargs,
    ):
        """Provide a data loader from portfolio.
        Parameters
        ----------
        collate_fn : Optional[Callable]
            The function to gather data molecules.
        assay : Union[None, str]
            Batch data from molecules using key provided to filter metadata.
        by : Union[Iterable, str]
        Returns
        -------
        torch.utils.data.DataLoader
            Resulting data loader.
        """
        if collate_fn is None:
            # provide default collate function
            collate_fn = self._batch

        return torch.utils.data.DataLoader(
            collection=self._items,
            collate_fn=partial(
                collate_fn,
                by=by,
            ),
            *args,
            **kwargs,
        )


class MoleculeCollection(Collection):
    
    def __init__(self, molecules: Optional[Iterable] = [], *args, **kwargs) -> None:
        
        if isinstance(molecules, Molecule):
            molecules = [molecules]

        molecules = list(molecules)
        super().__init__(molecules, *args, **kwargs)
        # assert all(isinstance(molecule, Molecule) for molecule in self.molecules)

    @property
    def molecules(self):
        return self._items

    @molecules.setter
    def molecules(self, value):
        assert all(isinstance(molecule, Molecule) for molecule in value)
        self._items = value

    def featurize_all(self):
        """ Featurize all molecules in collection. """
        (molecule.featurize() for molecule in self.molecules)
        return self

    def erase_annotations(self):
        """Erase the metadata. """
        for molecule in self.molecules:
            molecule.erase_annotations()
        return self

    def update_annotations(self, other_annotations=None):
        """Update the metadata. """
        for idx, _ in enumerate(self.molecules):
            if isinstance(other_annotations, list):
                self.molecules[idx].update_annotations(other_annotations[idx])
            elif isinstance(other_annotations, dict):
                self.molecules[idx].update_annotations(other_annotations)
            else:
                self.molecules[idx].update_annotations()
        return self

    @property
    def smiles(self):
        """Return the list of SMILE strings in the datset. """
        return [molecule.smiles for molecule in self.molecules]

    @property
    def mols(self):
        """Return the list of RDMol in the datset. """
        return [molecule.mol for molecule in self.molecules]

    @classmethod
    def load(
        cls,
        path: str,
        reactant_names: Optional[list],
        include_metadata=False
    ):
        """Read collection from pandas DataFrame.
        Parameters
        ----------
        TODO

        Examples
        --------
        >>> TODO
        >>> collection = load(path, ['reagsmi1', 'reagsmi2'])
        """
        # load from disk
        records = dg.datasets.disk_loader(path)
        
        def _make_mol(record):
            reactants = [
                Molecule(record.GetProp(r))
                for r in reactant_names
                if r in record.GetPropNames()
            ]
            if not reactants:
                reactants = [Molecule(record)]
            m = Molecule(record, reactants)
            return m
        
        molecules = [_make_mol(r) for r in records]
        return cls(molecules)
    
    def set_status(
        self,
        status: str,
        by: Optional[Callable] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Set the status of molecules in the self.molecules collection based on a filter.

        Parameters
        ----------
        status : str
            The new status to set for the molecule.
        by : callable, optional
            Determines if a molecule's status should be changed. Accept a dg.Molecule, returns a boolean.
        """
        for molecule in self.molecules:
            if by is None or by(molecule):
                molecule.status = status
                molecule.annotations[f'Step {status}'] = step

    @property
    def designed(self):
        return self.filter(lambda x: x.status == 'Designed')

    @property
    def scored(self):
        return self.filter(lambda x: x.status == 'Scored')

    @property
    def made(self):
        return self.filter(lambda x: x.status == 'Made')

    @property
    def tested(self):
        return self.filter(lambda x: x.status == 'Tested')

class ReactionCollection(Collection):
    
    def __init__(self, reactions: Optional[List] = [], *args, **kwargs) -> None:
        # assert all(isinstance(reaction, Reaction) for reaction in reactions)
        super().__init__(reactions, *args, **kwargs)
        
    @property
    def reactions(self):
        return self._items

    @property
    def names(self):
        """Return the list of reaction names in the datset. """
        return [reaction.name for reaction in self.reactions]

    @classmethod
    def from_json(
        cls,
        path: str,
        smarts_col: str,
        classes_col: Optional[str] = None
    ):
        """
        Load reactions.
        From SmilesClickChem: https://zenodo.org/record/4100676
        
        """

        # load from JSON
        reactions_df = pd.read_json(path).T.reset_index(drop=True)

        def _make_reaction(row):
            smirks = row[smarts_col]
            r = LazyReaction(smirks, id=row['reaction_name'])
            r = r.annotate_reactants(row[classes_col])
            return r

        reactions = reactions_df.apply(_make_reaction, axis=1).tolist()
        return cls(reactions)
