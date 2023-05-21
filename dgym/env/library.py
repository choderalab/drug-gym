import chemfp
import chemfp.arena
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from typing import Union, Iterable
from drug_gym.molecule import Molecule

def enumerate_library(
    hit,
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
    from copy import deepcopy

    products = []
    for i, route in enumerate(hit.synthetic_routes):

        reactants, reaction = deepcopy(route)

        # whatever is left after popping poised reactant are cognates
        poised_reactant = reactants.pop(poised_index)
        cognate_reactant = reactants[0]

        poised_reaction = reaction.poise(poised_index)
        cognate_class = poised_reaction.reactants[0].GetProp('class')

        # make sidechain_sets (assumes only two reactants)
        cognate_building_blocks = _filter(
            building_blocks,
            cognate_class,
            cognate_reactant.mol,
            size=size,
            sortby=sortby,
            fps=fps
        )
        sidechain_sets = [[poised_reactant.mol], cognate_building_blocks]

        # enumerate library
        library = AllChem.EnumerateLibraryFromReaction(
            poised_reaction.template,
            sidechain_sets,
            returnReactants=True
        )

        products.extend([
            Molecule(p.products[0], reactants=[Molecule(r) for r in p.reactants])
            for p in library
        ])


    return products


def _filter(
    building_blocks: dict,
    cognate_class: str,
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
    indices, bbs = building_blocks[cognate_class].values.T
    
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
    cognate_building_blocks = [_clean_mol(bb) for bb in bbs[argsort]]
    return cognate_building_blocks


def _fp_argsort(
    cognate_reactant: Chem.Mol,
    indices: Iterable[int],
    size: int = 10,
    fps: Union[chemfp.arena.FingerprintArena, None] = None,
) -> Iterable[int]:
    """
    TODO
    
    """
    def _remove_salts(mol):
        from rdkit.Chem.SaltRemover import SaltRemover
        remover = SaltRemover(defnData='[Cl,Br]')
        return remover.StripMol(mol)
    
    if fps is None:
        raise ValueError(
            'A `chemfp FingerprintArena` object must be provided'
            'if `sortby==\'fingerprint\'`.'
        )

    # clean salt in case it matters
    cognate_reagent_rdMol = _remove_salts(cognate_reactant)

    # sort by fingerprint
    argsort = chemfp.simsearch(
        k=size,
        query=Chem.MolToSmiles(cognate_reagent_rdMol),
        targets=fps.copy(indices=indices, reorder=False)
    ).get_indices()
    
    return argsort
