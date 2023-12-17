import random
import chemfp
import itertools
import dgym as dg
import numpy as np
import chemfp.arena
from rdkit import Chem
from itertools import chain, product
from rdkit.Chem import AllChem, DataStructs
from typing import Union, Iterable, Optional
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
        temperature: float
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
        reactants = self.generate_analogs(molecule.reactants, temperature)

        # Enumerate possible products given repertoire of reactions
        products = self.enumerate_products(reactants, size=size)

        # Add inspiration
        for product in products:
            product.inspiration = molecule
                
        return MoleculeCollection(products)


    def generate_analogs(self, original_molecules, temperature):
        """
        Returns a generator that samples analogs of the original molecules.
        """
        
        def _generate_analogs(samples, r=0):
            """
            A generator that efficiently yields analogs.

            """
            count = 0
            while True:
                
                # Gather constant and variable masks
                combo_indices = [i for i in range(len(samples[0]))]
                combos = self.random_combinations(combo_indices, r=r)
                constant_mask, variable_mask = next(combos)
                
                # Get variable reactants
                variable = samples[count, variable_mask].tolist()
                variable_reactants = [self.building_blocks[v] for v in variable]
                
                # Get constant reactants
                constant_reactants = [original_mols[c] for c in constant_mask]
                
                # Yield reactants
                reactants = [*constant_reactants, *variable_reactants]
                yield reactants
                
                # Increment
                count += 1

        if original_molecules:

            # Score analogs of each original reactant
            original_mols = [o.mol for o in original_molecules]
            analogs = self.get_analog_indices_and_scores(original_molecules)
            indices = list(analogs.iter_indices())
            scores = list(analogs.iter_scores())

            # Convert scores to probabilities
            probabilities = self.boltzmann(torch.tensor(scores), temperature)

            # Sample indices
            samples_idx = torch.multinomial(probabilities, 500)
            samples = torch.gather(torch.tensor(indices), 1, samples_idx).T
        
        else:
            samples = torch.multinomial(torch.ones([2, len(self.building_blocks)]), 500).T

        return _generate_analogs(samples)
    

    def get_analog_indices_and_scores(self, molecules):
        
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
        
        return chemfp.simsearch(
            queries = queries,
            targets = self.fingerprints,
            progress=False,
            k=500
        )

    @staticmethod
    def random_combinations(lst, r, k=500):
        """
        Yields a random combination of r items in list.
        """
        if r < 1:
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
    

    def enumerate_products(self, reactants, size):

        products = []
                    
        # Loop through permutations of reactants
        for reactants_ in reactants:
            for reactant_order in itertools.permutations(reactants_):
                for reaction in self.reactions:

                    # Check if completed enumeration
                    if len(products) >= size:
                        return products

                    # Verify reactants match
                    if len(reactants_) == len(reaction.reactants):

                        # Perform reaction
                        if output := reaction.run(reactant_order):
                            for product in output:
                                
                                # Check if valid molecule
                                if smiles := self.unique_sanitize(product):
                                    products += [Molecule(smiles, reactants = reactant_order)]
                        else:
                            continue
        
        return products

    def unique_sanitize(self, mol):
        
        # Sanitize
        smiles = Chem.MolToSmiles(mol[0])
        product = Chem.MolFromSmiles(smiles)
        
        # Check unique
        if product and smiles not in self.cache:
            self.cache.add(smiles)
            return smiles