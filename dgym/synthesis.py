import random
import chemfp
import chemfp.arena
import dgym as dg
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from typing import Union, Iterable
from dgym.molecule import Molecule
from dgym.reaction import Reaction


def enumerate_analogs(
    compound,
    repertoire,
    building_blocks,
    num_analogs: int,
    sortby: Union[str, dict] = {'fingerprint': 1.0, 'random': 0.0},
    fps: Union[chemfp.arena.FingerprintArena, None] = None,
) -> list:
    """
    Given a set of reagents, enumerate candidate molecules.

    Parameters
    ----------
    compound : dg.Molecule
        The hit from which analogs are enumerated.
    num_analogs : int
        Number of analogs to enumerate (for each compatible reaction).
    sortby : Union[str, dict]
        How to sort building blocks. Valid flags are 'fingerprint', 'random'.
        The values of the dictionary indicate the tradeoff.
        Default: 'fingerprint'.

    Returns
    -------
    all_products : list of enumerated products

    """
    # get matching reactions
    synthetic_routes = find_synthetic_routes(compound, repertoire)

    # enumerate poised synthetic library
    all_products = []
    for poised_index, _ in enumerate(compound.reactants):
        
        products = design_library(
            compound,
            synthetic_routes,
            building_blocks,
            index=poised_index,
            size=num_analogs,
            sortby=sortby,
            fps=fps
        )
        all_products.extend(products)

    return all_products



# Match molecular species and reactions.
# -----------------------------------------------

def find_synthetic_routes(
    compound,
    repertoire,
):
    """
    Find reactions compatible with a given compound.
    
    Parameters
    ----------
    hit : rdkit.Chem.rdchem.Mol
        The end-product of a reaction, comprising the hit.
    
    reagents : list of dict
        The synthetic reagents for the hit.
        Each dict has keys 'id', 'smiles', 'rdMol'.
    
    repertoire : pandas.DataFrame
        All available reactions.
            
    Returns
    -------
    A list of compatible reactions, each represented as a dictionary.

    """
    def _match_reaction(compound, reaction):
        
        if len(compound.reactants) != len(reaction.reactants):
            return None

        # reorder reactants
        for idx, _ in enumerate(reaction.reactants):
            poised_reaction = reaction.poise(idx)
            # test ordering for 1:1 match
            if all([reactant.has_substruct_match(template)
                for reactant, template
                in zip(compound.reactants, poised_reaction.reactants)
            ]):
                return poised_reaction
        
        return None

    synthetic_routes = []
    
    # loop through all reactions
    for reaction in repertoire:
        
        # find 1:1 matching ordering, if it exists
        reaction_match = _match_reaction(compound, reaction)
        
        if reaction_match:
            synthetic_routes.append(reaction_match)
    
    return synthetic_routes


# Enumerate compound library
# -----------------------------------------------

def design_library(
    compound: Molecule,
    synthetic_routes: Iterable[Reaction],
    building_blocks: dict,
    index: int = 0,
    size: int = 10,
    sortby: Union[str, dict] = {'fingerprint': 1.0, 'random': 0.0},
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
    def _clean(lst):
        return [Chem.MolFromSmiles(Chem.MolToSmiles(m)) for m in lst]
    
    def _remove_salts(m):
        from rdkit.Chem.SaltRemover import SaltRemover
        remover = SaltRemover(defnData='[Cl,Br]')
        return remover.StripMol(m.mol)

    def _fp_argsort(cognate_reactant: Chem.Mol, indices: Iterable[int],
                    size: int, fps: chemfp.arena.FingerprintArena) -> Iterable[int]:
        # remove salt in case it matters
        cognate_reactant = _remove_salts(cognate_reactant)
        return chemfp.simsearch(
            k=size, query=Chem.MolToSmiles(cognate_reactant),
            targets=fps.copy(indices=indices, reorder=False)).get_indices()

    products = []
    for reaction in synthetic_routes:
            
            # poise fragment and each reaction
            poised_compound = compound.poise(index)
            poised_reaction = reaction.poise(index)
            
            # filter building blocks compatible with poised fragment
            cognate_class = poised_reaction.reactants[1].GetProp('class')
            indices, building_blocks_subset = building_blocks[cognate_class].values.T
            
            # sort building blocks by fingerprint similarity and random
            argsort = []
            size_fp, size_rand = [int(v * size) for v in sortby.values()]
            if size_fp:
                argsort.extend(_fp_argsort(poised_compound.reactants[1], indices, size_fp, fps=fps))
            if size_rand:
                argsort.extend(random.sample(range(len(indices)), size_rand))
            cognate_building_blocks = _clean(building_blocks_subset[argsort])
            
            # enumerate library
            library = AllChem.EnumerateLibraryFromReaction(
                poised_reaction.template,
                [[poised_compound.reactants[0].mol], cognate_building_blocks],
                returnReactants=True
            )
            
            (p.UpdatePropertyCache() for p in library)
            for p in library:
                products.append(
                    Molecule(
                        p.products[0],
                        reactants=[Molecule(r) for r in p.reactants]
                    )
                )

    return products