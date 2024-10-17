from __future__ import annotations

import os
import math
import rdkit
import torch
import numpy as np
from typing import Union, Optional, Iterable
from collections import defaultdict
from scipy.special import logsumexp
from rdkit import Chem
from rdkit.Chem import Descriptors
from contextlib import contextmanager
from catboost import CatBoostRegressor
from meeko import MoleculePreparation, PDBQTWriterLegacy
from dgym.collection import MoleculeCollection
from sklearn.preprocessing import normalize
from scikit_mol.descriptors import MolecularDescriptorTransformer

import re
import glob
from itertools import islice

import subprocess
import os
import shutil
import tempfile

class OracleCache(dict):
    def __missing__(self, key):
        self[key] = float('nan')
        return float('nan')
    
    def copy(self):
        return OracleCache(self)

class Oracle:
    
    def __init__(self) -> None:
        self.cache = OracleCache()

    def __call__(self, molecules: Union[MoleculeCollection, list], **kwargs):

        # Normalize input            
        return_list = isinstance(molecules, Iterable)
        molecules = MoleculeCollection(molecules)

        # Make predictions
        predictions = self.get_predictions(molecules, **kwargs)

        return predictions if return_list else predictions[0]        
    
    def reset(self):
        self.cache = OracleCache()
        return self
    
    @contextmanager
    def suspend_cache(self):
        old_cache = self.cache.copy()
        self.reset()
        try:
            yield
        finally:
            self.cache = old_cache
    
    def get_predictions(
        self,
        molecules: Union[MoleculeCollection, list],
        use_cache: bool = True,
        **kwargs
    ):
        if use_cache:
            return self._get_predictions(molecules, **kwargs)
        else:
            with self.suspend_cache():
                return self._get_predictions(molecules, **kwargs)

    def _get_predictions(
        self,
        molecules: Union[MoleculeCollection, list],
        batch_size: Optional[int] = None,
        **kwargs
    ):
        # Normalize input
        if batch_size is None:
            batch_size = len(molecules)

        # Predict only uncached molecules and update cache
        uncached = lambda m: m.smiles not in self.cache
        if uncached_molecules := molecules.filter(uncached):
            for batch in uncached_molecules.batch(batch_size):
                smiles, predictions = self.predict(batch, **kwargs)
                self.cache.update(zip(smiles, predictions))

        # Match input ordering
        predictions = [self.cache[m.smiles] for m in molecules]
        
        return predictions

    def predict(self, molecules: MoleculeCollection):
        raise NotImplementedError
    
    def surrogate(self, sigma: float = 0.1) -> NoisyOracle:
        """
        Initialize a NoisyOracle surrogate model.
        """
        return NoisyOracle(self, sigma=sigma)


class NoisyOracle(Oracle):
    
    def __init__(self, oracle: Oracle, sigma: float = 0.1) -> None:
        """
        Initialize a NoisyOracle decorator for an Oracle instance.

        Parameters:
        - oracle (Oracle): The oracle instance to wrap.
        - sigma (float, optional): The standard deviation of the Gaussian noise to add to the oracle's predictions.
        """
        super().__init__()
        self.name = f'Noisy {oracle.name}'
        self.oracle = oracle
        self.sigma = sigma

    def get_predictions(
        self,
        molecules: Union[MoleculeCollection, list],
        **kwargs
    ):
        """
        Get predictions from the oracle and add Gaussian noise.

        Parameters:
        - molecules (Union[MoleculeCollection, list]): The molecules for which to predict values.
        - kwargs: Additional keyword arguments passed to the oracle's predict method.

        Returns:
        - List[float]: The noisy predictions for the given molecules.
        """
        # Utilize the wrapped oracle to get predictions
        predictions = self.oracle.get_predictions(molecules, **kwargs)

        # Add Gaussian noise to each prediction
        noisy_predictions = [p + np.random.normal(0, self.sigma) for p in predictions]

        return noisy_predictions

    def reset(self):
        """
        Resets the cache for both the NoisyOracle and the wrapped oracle.

        Returns:
        - NoisyOracle: self for method chaining.
        """
        super().reset()
        self.oracle.reset()
        return self


class CatBoostOracle(Oracle):
    
    def __init__(
        self,
        name: str,
        path: str
    ):
        super().__init__()
        self.name = name
        self.path = path
        self.regressor = CatBoostRegressor().load_model(path)
    
    def predict(
        self,
        molecules: MoleculeCollection,
        parallel: Union[bool, int] = True,
        **kwargs
    ):
        
        # Get SMILES
        smiles = [m.smiles for m in molecules]
        mols = [m.mol for m in molecules]

        # Score molecules
        X = self._featurize(mols, parallel=parallel)
        scores = self.regressor.predict(X)
        
        return smiles, scores
    
    def _featurize(
        self,
        rd_mols: Iterable,
        parallel: Union[bool, int] = True
    ):
        
        desc_list = [
            'ExactMolWt', 'FpDensityMorgan1',
            'FpDensityMorgan2', 'FpDensityMorgan3',
            'HeavyAtomMolWt', 'MaxAbsPartialCharge',
            'MaxAbsPartialCharge', 'MinAbsPartialCharge',
            'MinPartialCharge', 'MolWt',
            'NumRadicalElectrons', 'NumValenceElectrons',
            'MolLogP', 'FractionCSP3',
            'HeavyAtomCount', 'NHOHCount',
            'NOCount', 'NumAliphaticCarbocycles',
            'NumAliphaticHeterocycles', 'NumAliphaticRings',
            'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
            'NumAromaticRings', 'NumHAcceptors',
            'NumHDonors', 'NumHeteroatoms',
            'NumRotatableBonds', 'NumSaturatedCarbocycles',
            'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount',
        ]

        transformer = MolecularDescriptorTransformer(
            desc_list, parallel=parallel)
        X = transformer.transform(rd_mols)
        X = normalize(np.nan_to_num(X))

        return X

# class DGLOracle(Oracle):

#     def __init__(
#         self,
#         name: str,
#         mol_to_graph=dgllife.utils.MolToBigraph(
#             add_self_loop=True,
#             node_featurizer=dgllife.utils.CanonicalAtomFeaturizer()
#         )
#     ):
#         super().__init__()
#         self.name = name
#         self.mol_to_graph = mol_to_graph

#         # load model
#         # TODO - avoid internet download if the model is already local
#         # if f'{name}_pre_trained.pth' in os.listdir():
#         self.model = dgllife.model.load_pretrained(self.name, log=False)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
#         self.model.eval()
        
#     def predict(self, molecules: MoleculeCollection, **kwargs):
        
#         # featurize
#         graphs = [
#             self.mol_to_graph(m.update_cache().mol)
#             for m in molecules
#         ]
#         graph_batch = dgl.batch(graphs).to(self.device)
#         feats_batch = graph_batch.ndata['h']
        
#         # perform inference
#         scores = self.model(graph_batch, feats_batch).flatten().tolist()
#         smiles = [m.smiles for m in molecules]

#         return smiles, scores

class ReactionStepOracle(Oracle):

    def __init__(
        self,
        name: str,
        constant: Optional[Union[float, int]] = None,
    ):
        super().__init__()
        self.name = name
        self.constant = constant if constant else 1

    def predict(self, molecules: MoleculeCollection, **kwargs):
        smiles = [m.smiles for m in molecules]
        scores = [self.reaction_depth(m.dump()) for m in molecules]
        return smiles, scores

    def reaction_depth(self, synthetic_route):
        """
        Compute the depth of a reaction dictionary, representing the number of reaction steps
        necessary to make the molecule.
        
        Parameters
        ----------
        molecule : dg.molecule.Molecule
            The dictionary representing the reaction steps to produce a molecule.
        
        Returns
        -------
        int
            The depth of the reaction dictionary.
        """
        if 'reactants' not in synthetic_route or not synthetic_route['reactants']:
            return 0
        
        return 1 + max(self.reaction_depth(reactant) for reactant in synthetic_route['reactants'])

class ConstantOracle(Oracle):

    def __init__(
        self,
        name: str,
        constant: Optional[Union[float, int]] = None,
    ):
        super().__init__()
        self.name = name
        self.constant = constant if constant else 1

    def predict(self, molecules: MoleculeCollection, **kwargs):
        smiles = [m.smiles for m in molecules]
        scores = [self.constant for _ in molecules]
        return smiles, scores
    
class GaussianOracle(Oracle):
    
    def __init__(
        self,
        name: str,
        loc: Optional[Union[float, int]] = None,
        scale: Optional[Union[float, int]] = None,
    ):
        super().__init__()
        self.name = name
        self.loc = loc if loc else 0
        self.scale = scale if scale else 1

    def predict(self, molecules: MoleculeCollection, **kwargs):
        smiles = [m.smiles for m in molecules]
        scores = np.random.normal(self.loc, self.scale, len(molecules)).tolist()
        return smiles, scores

class RandomOracle(Oracle):

    def __init__(
        self,
        name: str
    ):
        super().__init__()
        self.name = name

    def predict(self, molecules: MoleculeCollection, **kwargs):
        smiles = [m.smiles for m in molecules]
        scores = np.random.random(len(molecules)).tolist()
        return smiles, scores


class RDKitOracle(Oracle):

    def __init__(
        self,
        name: str,
        descriptor: Optional[str] = None
    ):
        super().__init__()
        self.name = name

        # load descriptor
        if not descriptor:
            descriptor = self.name
        
        if hasattr(Descriptors, descriptor):
            self.descriptor = getattr(Descriptors, descriptor)
        elif descriptor == 'QED':
            self.descriptor = rdkit.Chem.QED.default

    def predict(self, molecules: MoleculeCollection, **kwargs):
        smiles = [m.smiles for m in molecules]
        scores = [self.descriptor(m.mol) for m in molecules]
        return smiles, scores


class DockingOracle(Oracle):

    def __init__(
        self,
        name: str,
        receptor_path: str,
        config: dict
    ):
        super().__init__()
        self.name = name
        self.receptor = receptor_path
        self.config = config

    def predict(
        self,
        molecules: MoleculeCollection,
        path: Optional[str] = None,
        units: Optional[str] = 'pIC50',
        **kwargs
    ):
        with self._managed_directory(path) as directory:

            # prepare ligands
            failed = self._prepare_ligands(molecules, directory)
            print(failed)

            # prepare command
            command = self._prepare_command(self.config, directory)
            print(command)

            # run docking
            resp = self._dock(command)
            print(resp)
            
            # gather results
            smiles, scores = self._gather_results(directory)

            # convert units
            scores = self._convert_units(scores, units)

        return smiles, scores
    

    def _convert_units(
        self,
        scores: list[float],
        units: Optional[str] = 'pIC50'
    ):
        match units:
            case 'deltaG':
                pass
            case 'pIC50':
                coef = -math.log10(math.e) / 0.5961473427254528
                scores = [coef * score for score in scores]
            case _:
                raise Exception(f'{units} is not a valid units. Must be `pIC50` or `deltaG`.')
        
        return scores


    def _gather_results(self, directory: str):

        scores = []
        smiles = []
        paths = glob.glob(f'{directory}/*_out.pdbqt')

        for idx, path in enumerate(paths):
            with open(path, 'r') as file:

                # extract SMILES from the file
                smiles_str = list(islice(file, 7))[-1]
                smiles_str = smiles_str.split(' ')[-1].split('\n')[0]
                smiles.append(smiles_str)

                # extract affinities
                file.seek(0)
                affinity_strs = [line for line in file if line.startswith('REMARK VINA RESULT')]
                process_affinity = lambda s: re.search(r'-?\d+\.\d+', s).group()
                energies = [float(process_affinity(a)) for a in affinity_strs]

                # compute boltzmann sum
                score = self._compute_deltaG(energies)
                # score = min(energies)

                # append to affinities
                scores.append(score)
        
        return smiles, scores
    
    def _dock(self, command: str):
        return subprocess.run(
            command,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            encoding='utf-8'
        )

    def _prepare_command(self, config, directory: str):
        
        # create inputs
        inputs = [
            '--receptor', self.receptor,
            '--ligand_index', os.path.join(directory, 'ligands.txt'),
            '--dir', directory,
        ]

        # add other parameters
        flags = [f'--{k}' for k in config.keys()]
        params = config.values()
        inputs.extend([
            str(elem)
            for pair in zip(flags, params)
            for elem in pair
        ])

        return ' '.join(['unidock', *inputs])

    def _prepare_ligands(self, molecules, directory: str):       
        failed = []
        paths = []
        for idx, mol in enumerate(molecules):
            
            # compute PDBQT
            try:
                pdbqt = self._get_pdbqt(mol)
                
                # write PDBQTs to disk
                path = os.path.join(
                    directory,
                    f'ligand_{idx}.pdbqt'
                )
                with open(path, 'w') as file:
                    file.write(pdbqt)
                paths.append(path)

                # write ligand text file
                ligands_txt = ' '.join(paths)
                path = os.path.join(directory, 'ligands.txt')
                with open(path, 'w') as file:
                    file.write(ligands_txt)
            except:
                failed.append(mol)

        return failed

    def _get_pdbqt(self, mol):

        # add hydrogens (without regard to pH)
        protonated_mol = rdkit.Chem.AddHs(mol.mol)

        # Generate 3D coordinates for the ligand.
        if rdkit.Chem.AllChem.EmbedMolecule(protonated_mol) != 0:
            raise ValueError("Failed to generate 3D coordinates for molecule.")

        # Check if the molecule has a valid conformer
        if protonated_mol.GetNumConformers() == 0:
            raise ValueError("No valid conformer found in the molecule.")

        # initialize preparation
        preparator = MoleculePreparation(rigid_macrocycles=True)
        setup = preparator.prepare(protonated_mol)[0]

        # write PDBQT
        pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup, bad_charge_ok=True)
        
        return pdbqt_string

    def _compute_deltaG(self, energies, temperature=298.15):

        # Boltzmann constant in kcal/(mol·K) multiplied by temperature in K
        kT = 1.987204259e-3 * temperature
        
        # Energies should be in kcal/mol
        energies = np.array(energies)
        
        # Calculate the overall Gibbs free energy
        boltz_sum = -kT * logsumexp(-energies / kT)
        
        return boltz_sum

    @contextmanager
    def _managed_directory(self, dir_path=None):
        is_temp_dir = False
        if dir_path is None:
            dir_path = tempfile.mkdtemp()
            is_temp_dir = True
        try:
            yield dir_path
        finally:
            if is_temp_dir:
                shutil.rmtree(dir_path)



# class NeuralOracle(Oracle):

#     def __init__(
#         self,
#         name: str,
#         state_dict_path: str = None,
#         config: Optional[dict] = None,
#     ):
#         super().__init__()
#         self.name = name

#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')

#         # load model architecture
#         model = self.model_factory(config=config)

#         # load model weights
#         self.model = self.load_state_dict(model, state_dict_path)

#     def predict(self, molecules: MoleculeCollection, **kwargs):

#         # make mol_to_graph util
#         mol_to_graph = dgllife.utils.MolToBigraph(
#             add_self_loop=True,
#             node_featurizer=dgllife.utils.CanonicalAtomFeaturizer()
#         )

#         # featurize
#         graphs = [
#             mol_to_graph(m.update_cache().mol)
#             for m in molecules
#         ]

#         # batch
#         graph_batch = dgl.batch(graphs).to(self.device)

#         # perform inference
#         with torch.no_grad():
#             preds = self.model({'g': graph_batch})

#         # clip to limit of detection
#         scores = torch.clamp(preds, 4.0, None).ravel().tolist()
#         smiles = [m.smiles for m in molecules]

#         return smiles, scores

#     def model_factory(self, config=None):
#         """
#         Build appropriate 2D graph model.

#         Parameters
#         ----------
#         config : Union[str, dict], optional
#             Either a dict or JSON file with model config options. If not passed,
#             `config` will be taken from `wandb`.

#         Returns
#         -------
#         mtenn.conversion_utils.GAT
#             GAT graph model
#         """
#         from dgllife.utils import CanonicalAtomFeaturizer
#         from mtenn.conversion_utils import GAT

#         # defaults
#         if not config:
#             config = {
#                 "dropout": 0.05,
#                 "gnn_hidden_feats": 64,
#                 "num_heads": 8,
#                 "alpha": 0.06,
#                 "predictor_hidden_feats": 128,
#                 "num_gnn_layers": 5,
#                 "residual": True
#             }


#         # config.update({"in_node_feats": CanonicalAtomFeaturizer().feat_size()})
#         in_node_feats = CanonicalAtomFeaturizer().feat_size()

#         model = GAT(
#             in_feats=in_node_feats,
#             hidden_feats=[config["gnn_hidden_feats"]] * config["num_gnn_layers"],
#             num_heads=[config["num_heads"]] * config["num_gnn_layers"],
#             feat_drops=[config["dropout"]] * config["num_gnn_layers"],
#             attn_drops=[config["dropout"]] * config["num_gnn_layers"],
#             alphas=[config["alpha"]] * config["num_gnn_layers"],
#             residuals=[config["residual"]] * config["num_gnn_layers"],
#         )

#         return model

#     def load_state_dict(self, model, state_dict_path):
#         """
#         Get the state dictionary for the neural net from disk.

#         Parameters
#         ----------
#         model : ...
#             The model object.

#         state_dict_path : str
#             The path on disk to the model weights.
        
#         Returns
#         -------
#         mtenn.conversion_utils.GAT
#             GAT graph model        
#         """
        
#         # get state dict from disk
#         preloaded_state_dict = torch.load(state_dict_path)
#         state_dict = dict(zip(
#             model.state_dict().keys(),
#             preloaded_state_dict.values()
#         ))

#         # load into model
#         model.load_state_dict(state_dict)
#         model.eval()
#         model = model.to(self.device)

#         return model