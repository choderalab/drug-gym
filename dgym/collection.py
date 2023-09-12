"""Collections and their serving."""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import dgym
import torch
import pandas as pd
from rdkit import Chem
from dgym.molecule import Molecule
from dgym.reaction import Reaction
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
    def __init__(self, items: Optional[List] = []) -> None:
        super(Collection, self).__init__()
        assert isinstance(items, List)
        
        self._items = items
        self._lookup = None

    def __repr__(self):
        return "{collection} with {size} {items}s".format(
            collection = self.__class__.__name__,
            size = len(self),
            items = self._items[0].__class__.__name__
        )

    def _construct_lookup(self):
        """Construct lookup table for molecules."""
        self._lookup = {item.name: item for item in self._items}

    @property
    def annotations(self):
        return pd.DataFrame([item.annotations for item in self._items])
    

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
        return item.name in self.lookup

    def filter(self, by: Callable):
        return self.__class__([item for item in self._items if by(item)])

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
        """Determin if two objects are identical."""
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
        * If the key is a string, return a collection of all molecules with
            this SMILES.
        * If the key is a molecule, extract the SMILES string and index by
            its SMILES.
        * If the key is a tensor, flatten it to treat it as a list.
        * If the key is a list, return a collection with molecules indexed by
            the elements in the list.
        * If the key is a slice, slice the range and treat at as a list.
        """
        if self._items is None:
            raise RuntimeError("Empty Collection.")
        if isinstance(key, int):
            return self._items[key]
        elif isinstance(key, str):
            return self.__class__([self.lookup[key]])
        elif isinstance(key, type(self._items[0])):
            return self.lookup[key.name]
        elif isinstance(key, torch.Tensor):
            key = key.detach().flatten().cpu().numpy().tolist()
        elif isinstance(key, Iterable):
            return self.__class__([self._items[_idx] for _idx in key])
        elif isinstance(key, slice):
            return self.__class__(molecules=self._items[key])
        else:
            raise RuntimeError("The slice is not recognized.")

    def shuffle(self, seed=None):
        """ Shuffle the collection and return it. """
        import random
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
        if isinstance(other, list):
            return self.__class__(self._items + other)
        elif isinstance(other, Collection):
            return self.__class__(self._items + other._items)
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

    def batch(items=None):
        raise NotImplementedError

    def unique(self):
        return self.__class__(list(set(self._items)))

    def clone(self, unique=False):
        """ Return a copy of self. """
        import copy
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
        from functools import partial

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
    
    def __init__(self, molecules: Optional[List] = []) -> None:
        
        assert all(isinstance(molecule, Molecule) for molecule in molecules)
        super().__init__(molecules)

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

    def update_annotations(self, other_annotations):
        """Update the metadata. """
        for molecule in self.molecules:
            molecule.update_annotations(other_annotations)
        return self

    @property
    def smiles(self):
        """Return the list of SMILE strings in the datset. """
        return [molecule.smiles for molecule in self.molecules]

    @classmethod
    def from_sdf(
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
        >>> collection = from_sdf(path, ['reagsmi1', 'reagsmi2'])
        """

        # load from disk
        records = Chem.SDMolSupplier(path)
        
        def _make_mol(record):
            reactants = [
                Molecule(record.GetProp(r))
                for r in reactant_names
                if r in record.GetPropNames()
            ]
            m = Molecule(record, reactants)
            return m
        
        molecules = [_make_mol(r) for r in records]
        return cls(molecules)

    def batch(self, *args, **kwargs):
        return super()._batch(self.molecules, *args, **kwargs)

    @staticmethod
    def _batch(
        molecules=None, by=['g', 'y'],
        **kwargs,
    ):
        """Batches molecules by provided keys.
        Parameters
        ----------
        molecules : list of molecules
            Defaults to all molecules in Collection if none provided.
        assay : Union[None, str]
            Filter metadata using assay key.
        by : Union[Iterable, str]
            Attributes of molecule on which to batch.
        Returns
        -------
        ret : Union[tuple, dgl.Graph, torch.Tensor]
            Batched data, in order of keys passed in `by` argument.
        """
        from collections import defaultdict
        ret = defaultdict(list)

        # guarantee keys are a list
        by = [by] if isinstance(by, str) else by

        # loop through molecules
        for molecule in molecules:

            for key in by:
                if key == 'g':
                    # featurize graphs
                    if not molecule.is_featurized():
                        molecule.featurize()
                    ret['g'].append(molecule.g)

                else:
                    m = molecule.metadata[key]
                    ret[key].append(m)

        # collate batches
        for key in by:
            if key == 'g':
                ret['g'] = dgl.batch(ret['g'])
            else:
                ret[key] = torch.tensor(ret[key])

        # return batches
        ret = (*ret.values(), )
        if len(ret) < 2:
            ret = ret[0]
        return ret


class ReactionCollection(Collection):
    
    def __init__(self, reactions: Optional[List] = []) -> None:
        assert all(isinstance(reaction, Reaction) for reaction in reactions)
        super().__init__(reactions)

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
        import pandas as pd

        # load from JSON
        reactions_df = pd.read_json(path).T.reset_index(drop=True)

        def _make_reaction(row):
            smirks = row[smarts_col]
            r = Reaction(smirks, name=row['reaction_name'])
            r = r.annotate_reactants(row[classes_col])
            return r

        reactions = reactions_df.apply(_make_reaction, axis=1).tolist()
        return cls(reactions)

