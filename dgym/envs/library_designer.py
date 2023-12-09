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
        num_reactants = self._get_num_reactants(num_analogs, temperature)
        reactants = [
            self._get_analogs(r, temperature, num_reactants)
            for r in molecule.reactants
        ]

        # Enumerate possible products given repertoire of reactions
        candidates = self._enumerate_products(reactants)

        # Add inspiration
        for candidate in candidates:
            candidate.inspiration = molecule
        
        if len(candidates) <= num_analogs:
            return candidates

        # Sample from product candidates
        products = self._sample_products(candidates, molecule, temperature, num_analogs)
        
        return MoleculeCollection(products)

    def _get_num_reactants(self, num_analogs, temperature):
        """
        Adjusts the number of analogs based on the temperature while ensuring the number
        of analogs does not fall below 5 and temperature is non-negative.

        Parameters:
        temperature (float): The current temperature. Assumed to be non-negative.
        original_analogs (int): The original number of analogs.

        Returns:
        int: The adjusted number of analogs based on the temperature, constrained to be no less than 5.
        """
        import math
        num_reactants = math.sqrt(num_analogs * 2)
        num_reactants /= (1 - min(0.2, max(0.0, temperature)))
        num_reactants = max(5, int(num_reactants))
        return num_reactants

    def _get_analogs(self, reactant, temperature, size):
        
        # Perform similarity search
        result = chemfp.simsearch(
            k = 500,
            query = reactant.smiles,
            targets = self.fingerprints
        )
        
        # Collect scores
        indices, scores = zip(*result.get_indices_and_scores())

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

    def _sample_products(self, candidates, molecule, temperature, size):

        # Get similarities of candidates to original molecule
        scores = [self._tanimoto_similarity(molecule, c) for c in candidates]
        
        # Sample boltzmann-adjusted probabilities
        products = self._boltzmann_sampling(candidates, scores, temperature, size)

        return products
    
    def _tanimoto_similarity(self, mol1, mol2):
        """
        Calculate the Tanimoto similarity between two molecules represented by their SMILES strings.

        Parameters
        ----------
        smiles1 : str
            The SMILES representation of the first molecule.
        smiles2 : str
            The SMILES representation of the second molecule.

        Returns
        -------
        float
            The Tanimoto similarity between the two molecules.
        """    
        # Generate Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1.mol, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2.mol, 2)

        # Calculate Tanimoto similarity
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)

        return similarity

    def _boltzmann_sampling(self, array, probabilities, temperature, size=1):
        """
        Perform sampling based on Boltzmann probabilities with a temperature parameter.

        Parameters:
        probabilities (list of float): Original probabilities derived from Tanimoto similarity.
        temperature (float): Temperature parameter controlling the randomness of the sampling.
        size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
        numpy.ndarray: Indices of the sampled elements.
        """
        assert len(array) == len(probabilities)

        # Avoid dividing by zero
        temperature += 1e-2
        
        # Adjust probabilities using the Boltzmann distribution and temperature
        adjusted_probs = np.exp(np.log(probabilities) / temperature)
        adjusted_probs /= np.sum(adjusted_probs)

        # Perform the sampling
        rng = np.random.default_rng()
        choices = rng.choice(array, size=size, p=adjusted_probs, replace=False)
        
        # Shuffle sample
        choices = rng.permutation(choices)
        
        return choices.tolist()
