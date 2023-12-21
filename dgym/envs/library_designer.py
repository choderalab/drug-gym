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
        reactants = self.generate_analogs(
            molecule,
            mode=mode,
            temperature=temperature
        )

        # Enumerate possible products given repertoire of reactions
        products = self.enumerate_products(reactants, size=size)

        # Add inspiration
        for product in products:
            product.inspiration = molecule
                
        return MoleculeCollection(products)


    def generate_analogs(
        self,
        molecule: Molecule,
        mode: Literal['analog', 'expand'] = 'analog',
        temperature: Optional[float] = 0.0
    ):
        """
        Returns a generator that samples analogs of the original molecules.
        """
        
        def _mask_analogs(analogs, r=1):
            """
            A generator that efficiently yields analogs.

            """
            while any(analogs):
                
                if mode == 'analog':
                    combo_indices = [i for i in range(len(analogs))]
                    combos = self.random_combinations(combo_indices, r=r)
                    constant_mask, variable_mask = next(combos)

                elif mode == 'expand':
                    constant_mask, variable_mask = [0], [0]

                # Get variable reactants
                variable_reactants = [analogs[i].pop(0) for i in variable_mask]
                
                # Get constant reactants
                constant_reactants = [original_molecules[c].mol for c in constant_mask]
                
                # Yield reactants
                reactants = [*constant_reactants, *variable_reactants]
                yield reactants

        if mode == 'analog':

            original_molecules = molecule.reactants

            # Identify analogs of each original reactant
            indices, scores = self.get_analog_indices_and_scores(original_molecules)
            analogs = []
            for indices_ in indices:
                analogs.append([self.building_blocks[i] for i in indices_.tolist()])

            # Add size similarity to score
            scores += self.size_similarity(analogs, original_molecules)

            # Convert scores to probabilities
            probabilities = self.boltzmann(scores, temperature)

            # Reorder analogs
            samples = torch.multinomial(probabilities, 100).tolist()
            for idx, (analogs_, samples_) in enumerate(zip(analogs, samples)):
                analogs[idx] = [analogs_[index] for index in samples_]

        elif mode == 'expand':

            original_molecules = [molecule]

            # Unbiased sample of building_blocks
            samples = torch.multinomial(
                torch.ones([1, len(self.building_blocks)]),
                100
            ).tolist()
            analogs = [self.building_blocks[i] for i in samples]

        return _mask_analogs(analogs)
    

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
        
        results = chemfp.simsearch(
            queries = queries,
            targets = self.fingerprints,
            progress=False,
            k=100
        )

        indices = torch.tensor(list(results.iter_indices()))
        scores = torch.tensor(list(results.iter_scores()))

        return indices, scores
    
    def size_similarity(self, analogs, references):
        """
        Using L1-norm of building blocks with original molecules
        """
        def _absolute_sigmoid(n):
            return 1 / (1 + abs(n))

        similarity = []
        for analogs_, reference in zip(analogs, references):
            reference_size = reference.mol.GetNumAtoms()
            similarity.append([
                _absolute_sigmoid(a.GetNumAtoms() - reference_size)
                for a in analogs_
            ])
        
        return torch.tensor(similarity)
    
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
        return products

    def unique_sanitize(self, mol):
        
        # Sanitize
        smiles = Chem.MolToSmiles(mol[0])
        product = Chem.MolFromSmiles(smiles)
        
        # Check unique
        if product and smiles not in self.cache:
            self.cache.add(smiles)
            return smiles