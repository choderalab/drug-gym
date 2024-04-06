import random
import chemfp
import itertools
import dgym as dg
import numpy as np
from numpy.random import randint
import sys
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
from functools import lru_cache
from itertools import combinations
from copy import deepcopy

class Generator:
    
    def __init__(
        self,
        building_blocks,
        fingerprints,
        sizes,
        path: str ='./out/'
    ) -> None:
        
        # TODO lazy-load fingerprints and sizes
        
        self.building_blocks = building_blocks
        self.fingerprints = fingerprints
        self.sizes = sizes
    
    def __call__(
        self,
        molecules: Optional[Union[Iterable[Molecule], Molecule, str]] = None,
        temperature: Optional[float] = 0.0,
        strict: bool = False,
        method: Literal['original', 'similar', 'random'] = 'original',
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Returns a generator that samples analogs of the original molecules.
        """
        # Normalize type
        return_list = isinstance(molecules, list)
        molecules = [molecules] if not return_list else molecules
        molecules = [Molecule(m) for m in molecules if m]
        
        if method == 'original' and molecules:
            generators = [itertools.repeat(m) for m in molecules]
        
        else:
            # Unbiased sample of indices if random
            if method == 'random' or not molecules:
                if seed: torch.manual_seed(seed)
                molecules = itertools.repeat(None)
                probabilities = torch.ones([1, len(self.building_blocks)])
                samples = torch.multinomial(probabilities, 10_000).tolist()

            elif method == 'similar':
                
                # Identify analogs of each original molecule
                scores = self.fingerprint_similarity(molecules)

                # Add size similarity to score
                scores += self.size_similarity(molecules)

                # Weighted sample of indices
                if temperature == 0.0:
                    samples = torch.topk(scores, len(self.building_blocks))[1].tolist()
                
                # TODO set random seed
                else:
                    probabilities = self.boltzmann(scores, temperature)
                    samples = torch.multinomial(probabilities, len(self.building_blocks)).tolist()

            generators = [
                self._generator_factory(sampler, molecule, strict=strict)
                for sampler, molecule in zip(samples, molecules)
            ]

        return generators if return_list else generators[0]
    
    @viewable
    def _generator_factory(self, sampler, original=None, strict=False):
        
        for index in sampler:
            building_block = self._get_building_block(index)
            yield building_block
    
    @lru_cache(maxsize=None)
    def _get_building_block(self, index, original=None, strict=False):
        if building_block := self.building_blocks[index]:
            if strict:
                building_block = self.substruct_match(building_block, original)
            return Molecule(building_block)
        
    
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
            threshold = 0.0,
        )

        scores = torch.tensor(results.to_csr().A)

        return scores
    
    def size_similarity(self, molecules):
        """
        Normalized L1-norm of building blocks with original molecules
        """
        original_sizes = torch.tensor([m.mol.GetNumAtoms() for m in molecules])
        l1_norm = self.sizes - original_sizes[:, None]
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
        temperature = max(temperature, 1e-2) # Ensure temperature is not too low
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
        molecule: Molecule = None,
        size: int = 1,
        method: Literal['similar', 'grow', 'random'] = 'similar',
        temperature: Optional[float] = 0.0,
        strict: bool = False,
    ) -> Iterable:
        """
        Run reactions based on the specified mode, returning a list of products.

        Parameters:
        - mode (str): The mode of operation ('replace' or 'grow').
        - molecule (Molecule): The molecule to react.
        - size (int): The desired number of products.
        - library_designer (LibraryDesigner): An instance of a library designer.
        - generator (Generator): A generator function for reactants.

        Returns:
        - list: A list of product molecules.
        """
        # Normalize input
        if isinstance(molecule, int):
            size = molecule
            molecule = None

        # Prepare reaction conditions
        if method == 'random' or molecule is None:
            reactions = self.reactions
            reactants = [{'method': 'random'}, {'method': 'random'}]
        elif method == 'grow':
            reactions = self.reactions
            reactants = [{'product': molecule.smiles}, {'method': 'random', 'seed': randint(sys.maxsize)}]
        elif method == 'similar':
            reactions = self.match_reactions(molecule)
            reactants = molecule.dump()['reactants']

        # Perform reactions
        products = OrderedSet()
        for reaction in reactions:
            reaction_tree = {'reaction': reaction.name, 'reactants': reactants}
            analogs = self.generate_analogs(reaction_tree, method=method)
            
            # Run reaction
            for analog in analogs:
                
                # Annotate metadata
                analog.inspiration = molecule
                if method == 'grow':
                    analog.reactants[0] = molecule

                # Collect products
                if len(products) < size:
                    if self.cache and analog in self._cache:
                        continue
                    products.add(analog)
                    self._cache.add(analog)
                else:
                    return products
                
        return products
    
    def match_reactions(self, molecule):
        """
        Finds the most specific reactions for the given molecule
        """
        if molecule.reaction and molecule.reaction in self.reactions:
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

    def generate_analogs(self, reaction_tree, method: str = 'similar'):
        """Initialize the reaction system with a random configuration variant."""
        
        # Make variant reaction trees
        num_annotations = 1 if method == 'similar' else 0
        variant_trees = self._annotate_reactants(
            reaction_tree, method=method, num_annotations=num_annotations)
        variant_products = [self._construct_reaction(v) for v in variant_trees]
        
        # Choose tree from which to yield product
        while True:
            chosen_products = random.choice(variant_products)
            yield from chosen_products
    
    def _annotate_reactants(self, reaction_tree, method: str = 'similar', num_annotations: int = 0):
        """
        Generates all unique variants of the reaction tree with the specified number of annotations applied.
        Utilizes deepcopy to ensure each variant is a completely separate copy.

        Parameters
        ----------
        reaction_tree : dict
            The initial reaction tree.
        mode : str
            The mode for annotation.
        num_annotations : int
            The number of annotations to apply.

        Returns
        -------
        list
            A list of all unique reaction tree variants with annotations applied.
        """
        paths = self._flatten_tree(reaction_tree)
        variants = []
        for combo in combinations(paths, num_annotations):
            new_tree = deepcopy(reaction_tree)
            for path in combo:
                new_tree = self._apply_annotation_to_path(new_tree, path, method)
            variants.append(new_tree)
        return variants

    def _flatten_tree(self, reaction_tree, path=()):
        """
        Flattens the reaction tree to a list of paths to each reactant.

        Parameters
        ----------
        reaction_tree : dict
            The reaction tree to flatten.
        path : tuple
            The current path in the tree, used for internal tracking.

        Returns
        -------
        list
            A list of paths to each reactant in the tree.
        """
        if 'reactants' in reaction_tree and reaction_tree['reactants']:
            paths = []
            for i, reactant in enumerate(reaction_tree['reactants']):
                paths.extend(self._flatten_tree(reactant, path + (i,)))
            return paths
        else:
            return [path]

    def _apply_annotation_to_path(self, reaction_tree, path, method):
        """
        Applies an annotation to a reactant specified by a path.
        This function now utilizes deepcopy to ensure modifications are isolated.

        Parameters
        ----------
        reaction_tree : dict
            The reaction tree.
        path : tuple
            The path to the reactant to be annotated.
        mode : str
            The annotation mode.

        Returns
        -------
        dict
            The reaction tree with the annotation applied.
        """
        if not path:
            return {'method': method, **reaction_tree}

        reactants = reaction_tree['reactants']
        for i, step in enumerate(path):
            if i == len(path) - 1:
                reactants[step] = self._apply_annotation_to_path(reactants[step], (), method)
            else:
                reactants = reactants[step]['reactants']

        return reaction_tree

    def _construct_reaction(self, reaction_tree):
        """
        Generates LazyReaction products based on a serialized reaction tree.
        
        Parameters:
        - serialized_tree: dict, the serialized reaction tree including SMILES strings.
        - generator: Generator object, used for generating molecules.
        
        Returns:
        - A generator for LazyReaction products.
        """
        # Base case: If tree is a simple molecule, return it appropriate generator
        if 'reactants' not in reaction_tree:
            product = reaction_tree.get('product', None)
            return self.generator(product, **reaction_tree)

        # Recursive case: Construct reactants and apply reaction
        if 'reaction' in reaction_tree \
            and 'reactants' in reaction_tree:
            reactants = [self._construct_reaction(reactant) for reactant in reaction_tree['reactants']]
            reaction = self.reactions[reaction_tree['reaction']]
            return reaction.run(reactants)

        raise Exception('`reaction_tree` must include a reaction or reactants.')