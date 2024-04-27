import random
import chemfp
import itertools
import dgym as dg
import numpy as np
from numpy.random import randint
import chemfp.arena
from rdkit import Chem
from itertools import chain, product
from rdkit.Chem import AllChem, DataStructs, rdDeprotect
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
import sys

MAX_INT = sys.maxsize

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
        search: Literal['fixed', 'similar', 'random'] = 'fixed',
        temperature: Optional[float] = 0.0,
        size_limit: Optional[int] = 1e3,
        strict: Optional[bool] = False,
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
        
        if search == 'fixed' and molecules:
            generators = [itertools.repeat(m) for m in molecules]
        
        else:
            # Unbiased sample of indices if random
            if search == 'random' or not molecules:
                if seed: torch.manual_seed(seed)
                molecules = itertools.repeat(None)
                
                # Filter by size
                valid_indices = torch.nonzero(self.sizes < size_limit).squeeze()
                
                # Sample randomly
                probabilities = torch.ones([1, len(valid_indices)])
                samples = valid_indices[torch.multinomial(probabilities, 100)].tolist()

            elif search == 'similar':
                
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
            building_block = self._get_building_block(
                index, original=original, strict=strict)
            yield building_block
    
    @lru_cache(maxsize=10_000)
    def _get_building_block(self, index, original=None, strict=False, deprotect=True):
        if building_block := self.building_blocks[index]:
            if strict:
                building_block = self.substruct_match(building_block, original)
            if deprotect:
                building_block = rdDeprotect.Deprotect(building_block)
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

        scores = torch.tensor(results.to_csr().toarray())

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

    def reset(self):
        self._cache = set()

    def design(
        self,
        molecule: Molecule = None,
        size: int = 1,
        strategy: Literal['replace', 'grow', 'random'] = 'replace',
        seed: Optional[int] = None,
        **kwargs
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
            strategy = 'random'
            size = molecule
            molecule = None

        # Prepare reaction conditions
        reactions = self.reactions.names if strategy != 'replace' \
            else self.match_reactions(molecule)
        if strategy == 'random' or not molecule:
            routes = [{'reactants': [{'search': 'random'}, {'search': 'random'}]}]
        elif strategy == 'grow':
            routes = [{'reactants': [{'search': 'fixed', 'product': molecule.smiles},
                                     {'search': 'random', 'size_limit': 10, 'seed': randint(MAX_INT)}]}]
        elif strategy == 'replace':
            routes = self._apply_annotations(molecule.dump(), annotations={'search': 'similar'}, **kwargs)

        # Perform reactions
        products = OrderedSet()
        for reaction in reactions:
            routes_ = [{'reaction': reaction, **r} for r in routes]

            # Run reaction
            analogs = self.generate_analogs(routes_)
            for analog in analogs:
                
                # Annotate metadata
                analog.inspiration = molecule
                if strategy == 'grow':
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
        # Normalize input
        if isinstance(molecule, dict):
            molecule = Molecule.load(molecule)
        
        # Return reaction if already annotated
        if molecule.reaction and molecule.reaction in self.reactions.names:
            return [molecule.reaction]

        # Filter by reactions compatible with reactants
        match_reactants = self.reactions.filter(
            lambda r: r.is_compatible(reactants=molecule.reactants))

        # Filter those reactions by compatibility with product
        match = match_reactants.filter(
            lambda r: r.is_compatible(product=molecule))
        
        return match.names if match \
            else (match_reactants.names if match_reactants else [])

    def generate_analogs(self, routes):
        """Initialize the reaction system with a random configuration variant."""
        
        # Derive pending reactions
        reaction_products = [self.construct_reaction(r) for r in routes]
        
        # Yield product from randomly chosen tree
        try:
            while True:
                chosen_products = random.choice(reaction_products)
                yield next(chosen_products)
        except StopIteration:
            return
    
    def _apply_annotations(
        self,
        route: dict,
        annotations: Optional[dict] = None,
        limit: Optional[int] = 1,
        **kwargs
    ):
        """
        Generates all unique variants of the synthetic route with the specified number of annotations applied.
        Utilizes deepcopy to ensure each variant is a completely separate copy.

        Parameters
        ----------
        route : dict
            The initial synthetic route.
        mode : str
            The mode for annotation.
        num_reactants : int
            The number of reactants to annotate in each combination.

        Returns
        -------
        list
            A list of all unique reaction tree variants with annotations applied.
        """
        # Get paths associated with each reactants
        paths = self._get_reactant_paths(route)

        # Normalize input
        limit = min(limit, len(paths))

        # Apply annotations
        reactant_variants = []
        for combo in combinations(paths, limit):
            new_tree = deepcopy(route)
            for path in combo:
                new_tree = self._apply_annotation_to_path(
                    new_tree, path, annotations, **kwargs)
            reactant_variants.append(new_tree)
        
        return reactant_variants

    def _get_reactant_paths(self, route, path=()):
        """
        Finds path to every reactant in synthetic route tree.

        Parameters
        ----------
        route : dict
            The synthetic route tree to flatten.
        path : tuple
            The current path in the tree, used for internal tracking.

        Returns
        -------
        list
            A list of paths to each reactant in the synthetic route.
        """
        if 'reactants' in route and route['reactants']:
            paths = []
            for i, reactant in enumerate(route['reactants']):
                paths.extend(self._get_reactant_paths(reactant, path + (i,)))
            return paths
        else:
            return [path]

    def _apply_annotation_to_path(self, route, path, annotations, **kwargs):
        """
        Applies an annotation to a reactant specified by a path.
        This function now utilizes deepcopy to ensure modifications are isolated.

        Parameters
        ----------
        route : dict
            The synethic route tree.
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
            return {**route, **annotations, **kwargs}

        # Recursive step
        reactants = route['reactants']
        for i, step in enumerate(path):
            if i == len(path) - 1:
                # Apply annotations at leaves
                reactants[step] = self._apply_annotation_to_path(
                    reactants[step], (), annotations, **kwargs)
            else:
                # Continue navigating path
                reactants = reactants[step]['reactants']

        return route

    def construct_reaction(self, route):
        """
        Generates LazyReaction products based on a serialized synthetic route.
        
        Parameters:
        - route: dict, the serialized synthetic route including SMILES strings.
        - generator: Generator object, used for generating molecules.
        
        Returns:
        - A generator for LazyReaction products.
        """
        # Base case: If tree is a simple molecule, return the appropriate generator
        if 'reactants' not in route:
            product = route.get('product', None)
            return self.generator(product, **route)

        # Recursive case: Construct reactants and apply reaction
        if 'reactants' in route:
            reactants = [self.construct_reaction(reactant) for reactant in route['reactants']]
            reaction_name = route['reaction'] if 'reaction' in route \
                else self.match_reactions(route)[0]
            reaction = self.reactions[reaction_name]
            return reaction.run(reactants)

        raise Exception('`route` must include a reaction or reactants.')