import os
import pickle
import itertools
from rdkit import Chem
from collections import defaultdict

__all__ = [
    'sort_fingerprints',
    'partition_building_blocks',
    'get_unique_reactants',
    'match_reactions'
]

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
            if bb.HasSubstructMatch(template['rdMol']):
                partitions[template['id']].append(
                    {'index': idx, 'rdMol': bb}
                )

    # convert records to dataframes
    partitions = {k: pd.DataFrame(v) for k, v in partitions.items()}
    
    return subsets


def get_unique_reactants(reactions):
    """
    Get all unique template classes from provided reactions.
    
    """
    unique_dict = {r.GetProp('class'): r
                   for reaction in reactions
                   for r in reaction.reactants}
    unique_classes = list(unique_dict.values())
    return unique_classes


# Match molecular species and reactions.
# -----------------------------------------------

def find_synthetic_routes(hit, reactions, ignore_product=False):
    """
    Find reactions compatible with a given hit.
    
    Parameters
    ----------
    hit : rdkit.Chem.rdchem.Mol
        The end-product of a reaction, comprising the hit.
    
    reagents : list of dict
        The synthetic reagents for the hit.
        Each dict has keys 'id', 'smiles', 'rdMol'.
    
    repertoire : pandas.DataFrame
        All available reactions.
    
    ignore_product : bool, default=False
        If False, require both the product and the reagents are compatible.
        If True, only require that the reagents are compatible.
        
    Returns
    -------
    A list of compatible reactions, each represented as a dictionary.

    """
    hit.synthetic_routes = []
    for reaction in reactions:

        # if the product is compatible
        if ignore_product or hit.has_substruct_match(reaction.products[0]):

            # find ordering that matches reactants to reactant templates
            compatible_reactants = _match_reactants(
                hit.reactants,
                reaction.reactants
            )

            # add synthetic route: tuple of reactants and reaction
            if compatible_reactants:
                synthetic_route = compatible_reactants, reaction
                hit.synthetic_routes.append(synthetic_route)
    
    return hit

def _match_reactants(reactants, templates):
    """
    Search for ordering with 1:1 match of hit reactants to reactant templates.
    
    """
    if len(reactants) != len(templates):
        return {}
    
    compatible_reactants = {}

    # for all possible reagent orderings
    for permutation in itertools.permutations(reactants):
        
        # if templates match elementwise
        if all([permutation[i].has_substruct_match(template)
                for i, template in enumerate(templates)]):
                        
            # return permutation
            compatible_reactants = list(permutation)
            break

    return compatible_reactants


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
