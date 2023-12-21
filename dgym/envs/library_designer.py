import random
import chemfp
import itertools
import dgym as dg
import numpy as np
import chemfp.arena
from rdkit import Chem
from itertools import chain, product
from rdkit.Chem import AllChem, DataStructs
from typing import Union, Iterable, Optional, Literal
from dgym.molecule import Molecule
from dgym.reaction import Reaction
from dgym.collection import MoleculeCollection
import torch


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
        molecule: Molecule,
        size: int,
        mode: Literal['analog', 'expand'] = 'analog',
        temperature: Optional[float] = 0.0
    ) -> Iterable:
        """
        Given a set of reagents, enumerate candidate molecules.

        Parameters
        ----------
        molecule : dg.Molecule
            The hit from which analogs are enumerated.
        num_analogs : int
            Number of analogs to enumerate (for each compatible reaction).

        Returns
        -------
        all_products : list of enumerated products

        """
        # Get analogs of the molecule reactants
        reactants = self.generate_reactants(
            molecule,
            mode=mode,
            temperature=temperature
        )

        # Get most specific possible reactions
        reactions = self.specify_reactions(molecule, mode=mode)

        # Enumerate possible products given repertoire of reactions
        products = self.enumerate_products(reactants, reactions, size=size)

        # Add inspiration
        for product in products:
            product.inspiration = molecule
        
        return MoleculeCollection(products)


    def generate_reactants(
        self,
        molecule: Molecule,
        mode: Literal['analog', 'expand'] = 'analog',
        temperature: Optional[float] = 0.0
    ):
        """
        Returns a generator that samples analogs of the original molecules.
        """
        
        def _mask_analogs(samples, r=1):
            """
            A generator that efficiently yields analogs.

            """
            while any(samples):
                
                if mode == 'analog':
                    combo_indices = [i for i in range(len(samples))]
                    combos = self.random_combinations(combo_indices, r=r)
                    constant_mask, variable_mask = next(combos)

                elif mode == 'expand':
                    constant_mask, variable_mask = [0], [0]

                # Get variable reactants
                variable = [samples[i].pop(0) for i in variable_mask]
                variable_reactants = [self.building_blocks[v] for v in variable]

                # Get constant reactants
                constant_reactants = [original_molecules[c] for c in constant_mask]
                
                # Yield reactants
                reactants = [*constant_reactants, *variable_reactants]
                yield reactants

        if mode == 'analog':

            original_molecules = molecule.reactants

            # Identify analogs of each original reactant
            indices, scores, sizes = self.fingerprint_similarity(original_molecules)

            # Add size similarity to score
            scores += self.size_similarity(original_molecules, sizes)

            # Convert scores to probabilities
            probabilities = self.boltzmann(scores, temperature)

            # Weighted sample of building blocks
            samples_idx = torch.multinomial(probabilities, 200)
            samples = torch.gather(indices, 1, samples_idx).tolist()

        elif mode == 'expand':

            original_molecules = [molecule]

            # Unbiased sample of building blocks
            probabilities = torch.ones([1, len(self.building_blocks)])
            samples = torch.multinomial(probabilities, 200).tolist()

        return _mask_analogs(samples)
    

    def fingerprint_similarity(self, molecules):
        
        fingerprint_type = self.fingerprints.get_fingerprint_type()
        fingerprints = [
            (m.id, fingerprint_type.from_smi(m.smiles))
            for m in molecules
        ]
        
        queries = chemfp.load_fingerprints(
            fingerprints,
            metadata = fingerprint_type.get_metadata(),
            reorder=False
        )
        
        results = chemfp.simsearch(
            queries = queries,
            targets = self.fingerprints,
            progress=False,
            k=500
        )

        indices = torch.tensor(list(results.iter_indices()))
        scores = torch.tensor(list(results.iter_scores()))

        get_size = lambda ids: [int(i.split(' ')[-1]) for i in ids]
        sizes = torch.tensor([get_size(ids) for ids in results.iter_ids()])

        return indices, scores, sizes
    
    def size_similarity(self, molecules, sizes):
        """
        Using L1-norm of building blocks with original molecules
        """
        original_sizes = torch.tensor([m.mol.GetNumAtoms() for m in molecules])
        l1_norm = sizes - original_sizes[:, None]
        return 1 / (1 + abs(l1_norm))
    
    @staticmethod
    def random_combinations(lst, r, k=100):
        """
        Yields a random combination of r items in list.
        """
        if r == 0:
            yield [], lst

        all_combinations = list(itertools.combinations(lst, r))
        selected_combinations = random.choices(all_combinations, k=k)
        for combo in selected_combinations:
            nonselected_items = tuple(item for item in lst if item not in combo)
            yield combo, nonselected_items

    @staticmethod
    def boltzmann(scores, temperature):
        """
        Applies the Boltzmann distribution to the given scores with a specified temperature.

        Parameters
        ----------
        scores : torch.Tensor
            The scores to which the Boltzmann distribution is applied.
        temperature : float
            The temperature parameter of the Boltzmann distribution.

        Returns
        -------
        torch.Tensor
            The probabilities resulting from the Boltzmann distribution.
        """
        temperature = max(temperature, 1e-2)  # Ensure temperature is not too low
        scaled_scores = scores / temperature
        probabilities = torch.softmax(scaled_scores, dim=-1)
        return probabilities
    
    def specify_reactions(self, molecule, mode):
        """
        Finds the most specific reactions for the given molecule
        """
        if mode == 'expand':
            return self.reactions
        
        if molecule.reaction in self.reactions:
            return [self.reactions[molecule.reaction]]

        # First, filter by reactions compatible with reactants
        match_reactants = [reaction for reaction in self.reactions
                           if reaction.is_compatible(reactants = molecule.reactants)]

        # Next, filter those reactions by compatibility with product
        if match_reactants_and_products := [reaction for reaction in match_reactants
                                            if reaction.is_compatible(molecule)]:
            reactions = dg.collection.ReactionCollection(match_reactants_and_products)

        elif match_reactants:
            reactions = dg.collection.ReactionCollection(match_reactants)

        else:
            reactions = self.reactions
        
        return reactions

    def enumerate_products(self, reactants, reactions, size):

        products = []
                    
        # Loop through permutations of reactants
        for reactants_ in reactants:
            for reactant_order in itertools.permutations(reactants_):
                for reaction in reactions:

                    # Check if completed enumeration
                    if len(products) >= size:
                        return products

                    # Perform reaction
                    if output := reaction.run(reactant_order):
                        for product in output:

                            # Check if valid molecule
                            if smiles := self.unique_sanitize(product):
                                products.append(
                                    Molecule(
                                        smiles,
                                        reactants = reactant_order,
                                        reaction = reaction
                                    )
                                )
        return products
    
    def unique_sanitize(self, mol):
        
        # Sanitize
        smiles = Chem.MolToSmiles(mol[0])
        product = Chem.MolFromSmiles(smiles)
        
        # Check unique
        if product and smiles not in self.cache:
            self.cache.add(smiles)
            return smiles