import random
import chemfp
import itertools
import dgym as dg
import chemfp.arena
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from typing import Union, Iterable, Optional
from dgym.molecule import Molecule
from dgym.reaction import Reaction


class LibraryDesigner:

    def __init__(
        self,
        reactions: list,
        building_blocks,
        fingerprints: Optional[chemfp.arena.FingerprintArena]
    ) -> None:

        self.reactions = reactions
        self.building_blocks = building_blocks
        self.fingerprints = fingerprints
        self.cache = set()
        # john: why not lazily recompute fingerprints only when needed, then cache it
        # for each object, what goes in, and what goes out

    def reset_cache(self):
        self.cache = set()

    def design(
        self,
        molecules: list,
        num_analogs: int,
        fraction_random: float
    ) -> list:
        """
        Given a set of reagents, enumerate candidate molecules.

        Parameters
        ----------
        molecule : dg.Molecule
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
        products = []
        for molecule in molecules:
            # get matching reactions
            reactions = self.find_compatible_reactions(molecule)
            # enumerate poised synthetic library
            analogs = self.enumerate_analogs(
                molecule,
                reactions,
                num_analogs=num_analogs,
                fraction_random=fraction_random
            )
            products.extend(analogs)
        return products

    def find_compatible_reactions(self, molecule) -> list[Reaction]:
        """
        Find reactions compatible with a given molecule.
        
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
        def _match_reaction(molecule, reaction):
            
            if len(molecule.reactants) != len(reaction.reactants): return None
            # reorder reactants, test 1:1 match
            for idx, _ in enumerate(reaction.reactants):
                poised_reaction = reaction.poise(idx)
                if all([
                    reactant.has_substruct_match(template)
                    for reactant, template
                    in zip(molecule.reactants, poised_reaction.reactants)
                ]): return poised_reaction
            return None

        synthetic_routes = []
        
        # loop through all reactions
        for reaction in self.reactions:
            
            # find 1:1 matching ordering, if it exists
            reaction_match = _match_reaction(molecule, reaction)
            
            if reaction_match:
                synthetic_routes.append(reaction_match)
        
        return synthetic_routes


    def enumerate_analogs(
        self,
        molecule: Molecule,
        reactions: list,
        num_analogs: int,
        fraction_random: float,
    ) -> list[Molecule]:
        """
        Returns enumerated product from poised reagent and compatible building blocks.

        Parameters
        ----------
        reaction : dict
            Reaction compatible with the hit.
        building_blocks : dict
            Building blocks partitioned by functional class.
        num_analogs : int
            Number of analogs to compute given a molecule.
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
                        size: int, fps: chemfp.arena.FingerprintArena):
            cognate_reactant = _remove_salts(cognate_reactant)
            return chemfp.simsearch(
                k=size, query=Chem.MolToSmiles(cognate_reactant),
                targets=fps.copy(indices=indices, reorder=False)
            ).get_indices()

        analogs = []
        for index, _ in enumerate(molecule.reactants):
            for reaction in reactions:

                # poise fragment and each reaction
                poised, cognate = molecule.poise(index).reactants
                reaction = reaction.poise(index)
                
                # filter building blocks compatible with poised fragment
                cognate_class = reaction.reactants[1].GetProp('class')
                indices, building_blocks_subset = self.building_blocks[cognate_class].values.T
                
                # sort building blocks by fingerprint similarity and random
                argsort = []
                size_rand = int(fraction_random * num_analogs)
                size_fp = num_analogs - size_rand
                if size_fp:
                    argsort.extend(_fp_argsort(cognate, indices, size_fp, fps=self.fingerprints))
                if size_rand:
                    argsort.extend(random.sample(range(len(indices)), size_rand))
                cognate_building_blocks = _clean(building_blocks_subset[argsort])

                # ensure no duplicates
                for c in cognate_building_blocks:
                    cognate_name = Chem.MolToSmiles(c)
                    combo = tuple([
                        f'{reaction.name}_{poised.name}_{cognate_name}',
                        f'{reaction.name}_{cognate_name}_{poised.name}'
                    ])
                    if combo in self.cache:
                        cognate_building_blocks.remove(c)
                    else:
                        self.cache.add(combo)
                
                # enumerate library
                library = AllChem.EnumerateLibraryFromReaction(
                    reaction.template,
                    [[poised.mol], cognate_building_blocks],
                    returnReactants=True
                )
                
                for p in library:
                    analog = Molecule(
                        p.products[0],
                        reactants=[Molecule(r) for r in p.reactants],
                        inspiration=molecule
                    ).update_cache()
                    analogs.append(analog)

        return analogs


# # Compute probabilities from logits to get probabilities
# probs = self.boltzmann(utility, self.temperature)

# @staticmethod
# def boltzmann(utility, temperature):
#     """Compute Boltzmann probabilities for a given set of utilities and temperature."""
#     energies = -np.array(utility)
#     exp_energies = np.exp(-energies / temperature)
#     return exp_energies / np.sum(exp_energies)
