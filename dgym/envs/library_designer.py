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

class AnalogGenerator:
    
    def __init__(
        self,
        building_blocks,
        fingerprints: chemfp.arena.FingerprintArena
    ) -> None:
        
        self.building_blocks = building_blocks
        self.fingerprints = fingerprints
    
    def __call__(
        self,
        molecules: Optional[Union[Iterable[Molecule], Molecule]] = None,
        temperature: Optional[float] = 0.0
    ):
        """
        Returns a generator that samples analogs of the original molecules.
        """
        def _make_generator(sampler):
            for index in sampler:
                yield Molecule(self.building_blocks[index])

        if isinstance(molecules, Molecule):
           molecules = [molecules]

        if molecules:

            # Identify analogs of each original molecule
            indices, scores, sizes = self.fingerprint_similarity(molecules)

            # Add size similarity to score
            scores += self.size_similarity(molecules, sizes)

            # Weighted sample of indices
            if temperature == 0.0:
                samples_idx = torch.argsort(scores, descending=True)
            
            else:
                probabilities = self.boltzmann(scores, temperature)
                samples_idx = torch.multinomial(probabilities, 200)

            samples = torch.gather(indices, 1, samples_idx).tolist()

        else:

            # Unbiased sample of indices
            probabilities = torch.ones([1, len(self.building_blocks)])
            samples = torch.multinomial(probabilities, 200).tolist()

        generators = [_make_generator(sampler) for sampler in samples]

        return generators

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
    
    def match_reactions(self, molecule):
        """
        Finds the most specific reactions for the given molecule
        """
        if molecule.reaction in self.reactions:
            return [molecule.reaction]

        # First, filter by reactions compatible with reactants
        match_reactants = [
            reaction
            for reaction in self.reactions
            if reaction.is_compatible(reactants = molecule.reactants)
        ]

        # Next, filter those reactions by compatibility with product
        match = [
            reaction
            for reaction in match_reactants
            if reaction.is_compatible(product = molecule)
        ]
        
        return match if match else (match_reactants if match_reactants else [])

    def enumerate_products(self, reactants, reactions, size):

        products = []
                    
        # Loop through permutations of reactants
        for reactants_ in reactants:
            for reactant_order in itertools.permutations(reactants_):
                for reaction in reactions:

                    # Check if completed enumeration
                    if len(products) >= size:
                        return products

                    if len(reactants_) == len(reaction.reactants):

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