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
from ..utils import viewable, OrderedSet
from collections.abc import Iterator

class Generator:
    
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
        temperature: Optional[float] = 0.0,
        strict: bool = False
    ):
        """
        Returns a generator that samples analogs of the original molecules.
        """
        return_list = isinstance(molecules, list)
        if molecules is None:

            # Unbiased sample of indices
            probabilities = torch.ones([1, len(self.building_blocks)])
            samples = torch.multinomial(probabilities, 200).tolist()
        
        else:
            
            if isinstance(molecules, Molecule):
                molecules = [molecules]

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

        generators = [
            self._generator_factory(sampler, molecule, strict=strict)
            for sampler, molecule in zip(samples, molecules)
        ]
        
        return generators if return_list else generators[0]

    @viewable
    def _generator_factory(self, sampler, original=None, strict=False):
        
        for index in sampler:
            building_block = self.building_blocks[index]
            
            if strict:
                building_block = self.substruct_match(building_block, original)
            
            if building_block:
                yield Molecule(building_block)
    
    def fingerprint_similarity(self, molecules):
        
        fingerprint_type = self.fingerprints.get_fingerprint_type()
        fingerprints = [
            (m.name, fingerprint_type.from_smi(m.smiles))
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
            progress = False,
            k = 500
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
    
    @staticmethod
    def substruct_match(new, old, protect=True):
        """
        Enforces old molecule is a substructure of new molecule.
        """
        if isinstance(new, Molecule):
            new = new.mol
        if isinstance(old, Molecule):
            old = old.mol

        if match := new.GetSubstructMatch(old):
            if protect:
                for atom in new.GetAtoms():
                    if atom.GetIdx() not in match:
                        atom.SetProp('_protected', '1')
            return new

class Designer:

    def __init__(
        self,
        generator: Generator,
        reactions: list,
        cache: bool = False,
    ) -> None:

        self.generator = generator
        self.reactions = reactions
        self.cache = cache
        self._cache = set()
        # john: why not lazily recompute fingerprints only when needed, then cache it
        # for each object, what goes in, and what goes out

    def reset_cache(self):
        self._cache = set()

    def design(
        self,
        molecule: Molecule,
        size: int,
        mode: Literal['analog', 'expand'] = 'analog',
        temperature: Optional[float] = 0.0,
        strict: bool = False,
        config: dict = {}, # TODO
    ) -> Iterable:
        """
        Run reactions based on the specified mode, returning a list of products.

        Parameters:
        - mode (str): The mode of operation ('analog' or 'expand').
        - molecule (Molecule): The molecule to react.
        - size (int): The desired number of products.
        - library_designer (LibraryDesigner): An instance of a library designer.
        - generator (Generator): A generator function for reactants.

        Returns:
        - list: A list of product molecules.
        """
        if mode == 'analog':
            reactions = self.match_reactions(molecule)
            random.shuffle(molecule.reactants) # TODO make a toggle
            reactants = molecule.reactants
            reactants = [
                reactants[0],
                self.generator(reactants[1], temperature=temperature, strict=strict)
            ]
            max_depth = None

        elif mode == 'expand':
            reactions = self.reactions
            reactants = [molecule, self.generator()]
            max_depth = 1

        # Perform reactions
        products = OrderedSet()
        for reaction in reactions:
            with molecule.set_reaction(reaction):
                with molecule.set_reactants(reactants):
                    
                    # Lazy load molecule analogs
                    analogs = self.retrosynthesize(molecule, max_depth=max_depth)

                    # Run reaction
                    for analog in analogs:
                        analog.inspiration = molecule
                        if len(products) < size:
                            if self.cache and analog in self._cache:
                                continue
                            products.add(analog)
                            self._cache.add(analog)
                        else:
                            return products
        
        return products

    def retrosynthesize(
        self,
        molecule,
        protect = False,
        max_depth = None,
        _depth = 0
    ):
        # Base cases
        if _depth == max_depth:
            return molecule

        if isinstance(molecule, Iterator):
            return molecule
        
        if not molecule.reactants:
            return molecule
        
        # Recursive case: Retrosynthesize each reactant
        retrosynthesized_reactants = [
            self.retrosynthesize(reactant, protect, max_depth, _depth + 1)
            for reactant in molecule.reactants
        ]

        if molecule.reaction is None:
            molecule.reaction = self.match_reactions(molecule)[0]

        # Use reaction to reconstruct the original molecule from its reactants
        output = molecule.reaction.run(retrosynthesized_reactants, protect=protect)
        
        return output
    
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
