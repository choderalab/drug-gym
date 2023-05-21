import chemfp
import chemfp.arena
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from typing import Union, Iterable

def enumerate(
    reaction,
    building_blocks: dict,
    poised_index: int = 0,
    size: int = 10,
    sortby: str = 'fingerprint',
    fps: Union[chemfp.arena.FingerprintArena, None] = None,
) -> list:
    """
    Returns enumerated product from poised reagent and compatible building blocks.

    Parameters
    ----------
    reaction : dict
        Reaction compatible with the hit.
    building_blocks : dict
        Building blocks partitioned by functional class.
    poised_index : int
        The index of the reagent to poise. Default: 0.
    size : int
        Size of library to enumerate.
    sortby : str
        How to sort building blocks. Valid flags are 'fingerprint', 'random'.
        Default: 'fingerprint'.
    fps : chemfp.arena.FingerprintArena, optional
        If sortby is set to 'fingerprint', fps must be provided.

    Returns
    -------
    Products of reactions. A list of `rdkit.Chem.Mol`.
    """
    # select a reaction and a reactant to poise
    poised_reaction = _poise_reaction(reaction, poised_index=poised_index)

    # poised reactant will always be leftmost
    reactants = poised_reaction['reactants']
    poised_reactant = reactants[0]
    cognate_reactants = reactants[1:]

    # make sidechain_sets
    poised_reagent = poised_reactant['reagent']['rdMol']
    cognate_building_blocks = _filter(
        building_blocks,
        cognate_reactants[0],
        size=size,
        sortby=sortby,
        fps=fps
    )
    sidechain_sets = [[poised_reagent], cognate_building_blocks]

    # enumerate library
    rxn = poised_reaction['rdReaction']
    library = AllChem.EnumerateLibraryFromReaction(rxn, sidechain_sets)
    prods = [next(library)[0] for _ in range(size)]
    
    return prods
            

def _poise_reaction(
    reaction: dict,
    poised_index: int = 0,
) -> dict:
    """
    Sorts reactants in `reaction` according to `poised_idx`.

    Parameters
    ----------
    reaction : dict
        Reaction compatible with the hit.
    poised_index : int
        The index of the reagent to poise. Default: 0.

    Returns
    -------
    A reaction dictionary with reactants sorted by `poised_idx`.
    """
    from copy import deepcopy
    
    def _move_idx_to_first(lst, idx):
        lst.insert(0, lst.pop(idx))
        return lst
    
    # don't modify in place
    reaction = deepcopy(reaction)
    
    # reorder according to poised index
    reaction['reactants'] = _move_idx_to_first(
        reaction['reactants'], poised_index
    )

    # make new Reaction SMARTS
    reactant_smarts = '.'.join([r['template']['smarts']
                                for r in reaction['reactants']])
    product_smarts = reaction['product']['smarts']
    poised_smarts = reactant_smarts + '>>' + product_smarts

    # convert to rdReaction
    reaction['rdReaction'] = AllChem.ReactionFromSmarts(poised_smarts)

    return reaction


def _filter(
    building_blocks: dict,
    cognate_reactant: dict,
    size: int = 10,
    sortby: str = 'fingerprint',
    fps: Union[chemfp.arena.FingerprintArena, None] = None,
) -> list:
    """
    Get building blocks corresponding to a cognate reactant.
    
    Parameters
    ----------
    subsets : dict
        Building blocks organized by functional class subsets.
    cognate_reactant : dict
        Cognate to the poised reagent.
    size : int
        Number of building blocks to retrieve.
    sortby : str
        How to sort building blocks. Valid flags are 'fingerprint', 'random'.
        Default: 'fingerprint'.
    fps : chemfp.arena.FingerprintArena, optional
        If sortby is set to 'fingerprint', fps must be provided.
    
    Returns
    -------
    list : cognate reagents from building blocks
    """
    def _clean_mol(mol):
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    # filter building blocks by cognate template
    cognate_template_id = cognate_reactant['template']['id']
    indices, bbs = building_blocks[cognate_template_id].values.T
    
    # sort building blocks
    if sortby == 'fingerprint':
        argsort = _fp_argsort(cognate_reactant, indices, size, fps=fps)
    elif sortby == 'diverse':
        pass
        # TODO
    elif sortby == 'random':
        import random
        argsort = random.sample(range(len(indices)), size)        

    # get cognate building blocks
    cognate_reagents = [_clean_mol(bb) for bb in bbs[argsort]]
    return cognate_reagents


def _fp_argsort(
    cognate_reactant: dict,
    indices: Iterable[int],
    size: int = 10,
    fps: Union[chemfp.arena.FingerprintArena, None] = None,
) -> Iterable[int]:
    """
    TODO
    
    """
    def _remove_salts(mol):
        from rdkit.Chem.SaltRemover import SaltRemover
        remover = SaltRemover(defnData="[Cl,Br]")
        return remover.StripMol(mol)
    
    if fps is None:
        raise ValueError(
            'A `chemfp FingerprintArena` object must be provided if `sortby==\'fingerprint\'`.'
        )

    # clean salt in case it matters
    cognate_reagent_rdMol = _remove_salts(
        cognate_reactant['reagent']['rdMol']
    )
    
    # sort by fingerprint
    argsort = chemfp.simsearch(
        k=size,
        query=Chem.MolToSmiles(cognate_reagent_rdMol),
        targets=fps.copy(indices=indices, reorder=False)
    ).get_indices()
    
    return argsort
