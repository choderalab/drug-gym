import os
import json
import pickle
import itertools
import pandas as pd
import psutil
from rdkit import Chem
from collections import defaultdict
from functools import wraps
from inspect import ismethod
from typing import Iterable
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, Draw

import chemfp
chemfp.set_license('20231114-columbia.edu@DAAABLGMDNEEHFALIFOLIONPFHFDJDOLHABF')
__all__ = [
    'sort_fingerprints',
    'partition_building_blocks',
    'get_unique_reactants',
    'match_reactions',
    'align_lineage'
]

# Building block processing.
# -----------------------------------------------
def compute_fingerprints(sdf_path: str = './', out_path: str = './out'):
    
    # TODO - make this work with SDF file path or SDFSupplier
    path = "./Enamine_Building_Blocks_Stock_262336cmpd_20230630.sdf"

    # define fingerprint encoder
    fp_type = chemfp.get_fingerprint_type("OpenBabel-ECFP4/1", {"nBits": 1024})
    fp_iterator = fp_type.read_molecule_fingerprints(path, id_tag='ID')

    # open output file
    with chemfp.open_fingerprint_writer(
        './Enamine_Building_Blocks_Stock_262336cmpd_20230630.fpb',
        metadata=fp_type.get_metadata(),
        reorder=False
    ) as writer:
        
        for idx, (id, fp) in enumerate(tqdm(fp_iterator)):
            writer.write_fingerprint(id, fp)


# Plotting.
# -----------------------------------------------
def align_lineage(molecule):
    """
    Recursively aligns a molecule to its chain of inspirations, going back to the first molecule.

    Parameters
    ----------
    molecule : dg.Molecule

    Returns
    -------
    list
        A list of aligned RDKit molecule objects, starting from the origin.
    """
    # Get molecule lineage
    lineage = molecule.lineage
    
    # Generate 3D coordinates for all molecules if not already present
    for molecule in lineage:
        mol = molecule.mol
        AllChem.Compute2DCoords(mol)

    # Align each molecule with its inspiration
    for i in range(1, len(lineage)):
        current_mol = lineage[i].mol
        inspiration_mol = lineage[i-1].mol

        # Find MCS between current molecule and its inspiration
        mcs_result = rdFMCS.FindMCS([current_mol, inspiration_mol])
        mcs_smarts = mcs_result.smartsString
        if not mcs_smarts:
            continue  # Skip if no MCS is found

        mcs_mol = Chem.MolFromSmarts(mcs_smarts)

        # Align the molecule to its inspiration based on MCS
        match1 = current_mol.GetSubstructMatch(mcs_mol)
        match2 = inspiration_mol.GetSubstructMatch(mcs_mol)
        if match1 and match2:
            atom_map = list(zip(match1, match2))
            AllChem.AlignMol(current_mol, inspiration_mol, atomMap=atom_map)

    return [molecule for molecule in lineage]

def draw(hit, reaction, prods, rowsize=3):
    """
    TODO
    
    """
    poised_group = reaction['reactants'][0]['template']['id']
    variable_group = reaction['reactants'][1]['template']['id']
    
    display(hit)
    print(f'Poised:\033[0m the \033[1m\033[91m{poised_group}\033[0m group.'
          f'\nVaried:\033[0m the \033[1m\033[94m{variable_group}\033[0m group.'
    )

    try:
        display(Chem.Draw.MolsToGridImage(
            prods,
            molsPerRow=rowsize,
            subImgSize=(300, 300)
        ))
    except:
        for p in prods:
            display(p)


# General utility
# -----------------------------------------------
def print_memory_usage():
    try:
        # Getting memory usage information
        memory = psutil.virtual_memory()
        # Converting bytes to a more human-readable form (GB)
        memory_in_gb = memory.used / (1024 ** 3)
        print(f"Memory Used: {memory_in_gb:.2f} GB")
        memory_in_gb = memory.available / (1024 ** 3)
        print(f"Memory Free: {memory_in_gb:.2f} GB")
    except Exception as e:
        # Handling possible exceptions that could occur during memory retrieval
        print(f"An error occurred: {e}")

def normalize_list(object):
    return object if isinstance(object, list) else [object]

def apply_recursive(object, function):
    """
    Apply a function to elements in a nested object structure if they are of a specific type.

    Parameters
    ----------
    obj : any
        The object to be traversed and modified.
    func : function
        The function to apply to elements of the target type.
    target_type : type
        The type of elements to which the function should be applied.

    Returns
    -------
    any
        The modified object with the function applied to elements of the target type.
    """
    try:
        return function(object)
    except:
        if isinstance(object, Iterable):
            return [apply_recursive(item, function) for item in object]

    return object

def serialize_with_class_names(data):
    """
    Recursively serializes a data structure, replacing non-JSON serializable objects
    with their class names as strings. This function supports nested dictionaries
    and lists.

    Parameters
    ----------
    data : dict or list
        The data structure to process for JSON serialization. Can be a dictionary
        or a list containing nested dictionaries and lists.

    Returns
    -------
    dict or list
        A new data structure with non-serializable objects replaced by their
        class names in string format.

    Examples
    --------
    >>> class CustomClass:
    ...     pass
    >>> data = {
    ...     'int': 1,
    ...     'list': [1, 2, 3, CustomClass()],
    ...     'dict': {'key': 'value', 'nested': {'custom_obj': CustomClass()}},
    ... }
    >>> result = serialize_with_class_names(data)
    >>> print(result)
    {'int': 1, 'list': [1, 2, 3, 'CustomClass'], 'dict': {'key': 'value', 'nested': {'custom_obj': 'CustomClass'}}}
    """
    if isinstance(data, dict):
        return {k: serialize_with_class_names(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_with_class_names(item) for item in data]
    else:
        try:
            json.dumps(data)
            return data  # Data is serializable, return as is
        except TypeError:
            return data.__class__.__name__  # Replace with class name if not serializable


class OrderedSet:

    def __init__(self, iterable = []):
        self.elements = []
        self.set = set()

        for element in iterable:
            self.add(element)

    def add(self, element):
        """Add an element to the ordered set.

        Parameters
        ----------
        element : Any
            The element to be added to the set.

        Returns
        -------
        None
        """
        if element not in self.set:
            self.elements.append(element)
            self.set.add(element)

    def remove(self, element):
        """Remove an element from the ordered set if it exists.

        Parameters
        ----------
        element : Any
            The element to be removed from the set.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the element is not in the set.
        """
        if element in self.set:
            self.elements.remove(element)
            self.set.remove(element)
        else:
            raise KeyError(f"Element {element} not found in OrderedSet.")

    def __iter__(self):
        """Return an iterator for the ordered set.

        Returns
        -------
        Iterator
            An iterator over the elements of the ordered set.
        """
        return iter(self.elements)

    def __contains__(self, element):
        """Check if an element is in the ordered set.

        Parameters
        ----------
        element : Any
            The element to check for in the set.

        Returns
        -------
        bool
            True if the element is in the set, False otherwise.
        """
        return element in self.set

    def __len__(self):
        """Return the number of elements in the ordered set.

        Returns
        -------
        int
            The number of elements in the set.
        """
        return len(self.set)

    def __repr__(self):
        """Return the string representation of the ordered set.

        Returns
        -------
        str
            The string representation of the ordered set.
        """
        return f"OrderedSet({self.set})"
    
    def __getitem__(self, index):
        return self.elements[index]


# Generators
# -----------------------------------------------
def viewable(func):
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if ismethod(func):
            # If the function is a bound method, pass 'self' separately
            return ViewableGenerator(args[0], func, *args[1:], **kwargs)
        else:
            # If the function is a standalone function or staticmethod
            return ViewableGenerator(None, func, *args, **kwargs)
    return wrapper

class ViewableGenerator:

    def __init__(self, instance, func, *args, **kwargs):
        """
        Initializes the GeneratorWrapper with an optional class instance, function/method, and arguments.

        Parameters
        ----------
        instance : object or None
            The instance of the class where the original method is defined, or None for standalone functions.
        func : callable
            The original generator-producing function, method, or staticmethod.
        args : tuple
            Arguments to pass to the function or method.
        kwargs : dict
            Keyword arguments to pass to the function or method.
        """
        self.instance = instance
        self.func = func
        self.args = args
        self.kwargs = kwargs
        if self.instance is not None:
            self.generator = self.func(self.instance, *self.args, **self.kwargs)
        else:
            self.generator = self.func(*self.args, **self.kwargs)

    def view(self):
        """
        Returns a generator that yields elements from the internal list.

        Yields
        ------
        element
            The next element in the list.
        """
        self.generator, generator_view = itertools.tee(self.generator)
        return generator_view
    
    def __iter__(self):
        """
        Returns the iterator for the generator.
        """
        return self.generator

    def __next__(self):
        """
        Returns the next item from the generator.
        """
        return next(self.generator)

    def __repr__(self):
        return f'Viewable: {self.generator.__repr__()}'
