import os
import pickle
import itertools
import pandas as pd
from rdkit import Chem
from collections import defaultdict
from functools import wraps
from inspect import ismethod

__all__ = [
    'sort_fingerprints',
    'partition_building_blocks',
    'get_unique_reactants',
    'match_reactions'
]

import chemfp
chemfp.set_license('20231114-columbia.edu@DAAABLGMDNEEHFALIFOLIONPFHFDJDOLHABF')


# Sort fingerprints according to building blocks.
# ---------------------------------

def sort_fingerprints(fps, building_blocks):
    """
    Align building block library to pre-computed fingerprints.
    
    """
    # get the argsort for the fingerprints
    sorted_indices = [
        fps.get_index_by_id(e.GetProp('ID'))
        if e is not None else 0
        for e in building_blocks
    ]

    # use argsort to reorder fingerprints
    return fps.copy(
        indices=sorted_indices,
        reorder=False
    )


# Organize building blocks by functional class.
# -----------------------------------------------

def partition_building_blocks(
    building_blocks=None,
    templates=None,
    out_dir='.'
):
    """
    Get partition building blocks according to provided functional groups.

    """    
    path = f'{out_dir}/out/building_block_subsets.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            building_blocks = pickle.load(handle)
    else:
        # compute assignments
        building_blocks = _partition_building_blocks(building_blocks, templates)
        
        # write to disk
        with open(path, 'wb') as handle:
            pickle.dump(building_blocks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return building_blocks


def _partition_building_blocks(building_blocks, templates):
    """
    Organize building block subsets by functional class.
    
    """
    partitions = defaultdict(list)
    
    # for each building block
    for idx, bb in enumerate(building_blocks):

        # weirdly, some library entries are empty
        if bb is None:
            continue

        # check substructure match with templates
        for template in templates:
            if bb.HasSubstructMatch(template):
                partitions[template.GetProp('class')].append(
                    {'index': idx, 'rdMol': bb}
                )

    # convert records to dataframes
    partitions = {k: pd.DataFrame(v) for k, v in partitions.items()}
    
    return partitions


def get_unique_reactants(reactions):
    """
    Get all unique template classes from provided reactions.
    
    """
    unique_dict = {r.GetProp('class'): r
                   for reaction in reactions
                   for r in reaction.reactants}
    unique_classes = list(unique_dict.values())
    return unique_classes



# Plotting.
# -----------------------------------------------

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
    
    # @staticmethod
    # def __call__(*args, **kwargs):
    #     return self.generator(*args, **kwargs)

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