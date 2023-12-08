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
        num_analogs: int,
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
        reactants = [
            self._get_analogs(r, num_analogs // 4, temperature)
            for r in molecule.reactants
        ]

        # Enumerate possible products given repertoire of reactions
        products = self._enumerate_products(reactants)
        
        return MoleculeCollection(products)

    def _get_analogs(self, reactant, size, temperature):
        
        # Perform similarity search
        indices, scores = zip(*chemfp.simsearch(
            k = 500,
            query = reactant.smiles,
            targets = self.fingerprints
        ).get_indices_and_scores())

        # Resample indices
        indices = self._boltzmann_sampling(indices, scores, temperature, size)

        # Get analogs
        analogs = [self.building_blocks[i] for i in indices]
        
        return analogs

    def _enumerate_products(self, analogs):

        def _bi_product(l1, l2):
            return chain(product(l1, l2), product(l2, l1))

        def _sanitize(mol):
            smiles = Chem.MolToSmiles(mol[0][0])
            return Chem.MolFromSmiles(smiles), smiles
        
        products = []
        for reactants in _bi_product(*analogs):
            for reaction in self.reactions:
                
                # Run reaction
                try:
                    prod = reaction.run(reactants)
                    prod, smiles = _sanitize(prod)
                except:
                    continue

                # Process product, check cache
                if prod and smiles not in self.cache:
                    self.cache.add(smiles)
                    products += [Molecule(smiles, reactants=reactants)]
        return products

    def _boltzmann_sampling(self, arr, probabilities, temperature, size=1):
        """
        Perform sampling based on Boltzmann probabilities with a temperature parameter.

        Parameters:
        probabilities (list of float): Original probabilities derived from Tanimoto similarity.
        temperature (float): Temperature parameter controlling the randomness of the sampling.
        size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
        numpy.ndarray: Indices of the sampled elements.
        """
        # Avoid dividing by zero
        temperature += 1e-2
        
        # Adjust probabilities using the Boltzmann distribution and temperature
        adjusted_probs = np.exp(np.log(probabilities) / temperature)
        adjusted_probs /= np.sum(adjusted_probs)

        # Perform the sampling
        rng = np.random.default_rng()
        choices = rng.choice(arr, size=size, p=adjusted_probs, replace=False).tolist()

        return choices
