from __future__ import annotations

import os
import dgl
import rdkit
import torch
import dgllife
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Union
from meeko import PDBQTWriterLegacy
from meeko import MoleculePreparation
from dgym.collection import MoleculeCollection

class Oracle:
    
    def __init__(self) -> None:
        self.cache = {}

    def __call__(self, molecules: Union[MoleculeCollection, list]):
        return self.get_predictions(molecules)
    
    def get_predictions(self, molecules: Union[MoleculeCollection, list]):

        if isinstance(molecules, list):
            molecules = MoleculeCollection(molecules)

        # identify uncached molecules
        not_in_cache = lambda m: m.smiles not in self.cache
        
        if uncached_molecules := molecules.filter(not_in_cache):
            
            # make predictions
            preds = self.predict(uncached_molecules)

            # cache results
            self.cache.update(zip(uncached_molecules.smiles, preds))

        # fetch all results (old and new) from cache
        return [self.cache[m.smiles] for m in molecules]

    def predict(self, molecules: MoleculeCollection):
        raise NotImplementedError



class DGLOracle(Oracle):

    def __init__(
        self,
        name: str,
        mol_to_graph=dgllife.utils.MolToBigraph(
            add_self_loop=True,
            node_featurizer=dgllife.utils.CanonicalAtomFeaturizer()
        )
    ):
        super().__init__()
        self.name = name
        self.mol_to_graph = mol_to_graph

        # load model
        # TODO - avoid internet download if the model is already local
        # if f'{name}_pre_trained.pth' in os.listdir():
        self.model = dgllife.model.load_pretrained(self.name, log=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict(self, molecules: MoleculeCollection):
        
        # featurize
        graphs = [
            self.mol_to_graph(m.update_cache().mol)
            for m in molecules
        ]
        graph_batch = dgl.batch(graphs).to(self.device)
        feats_batch = graph_batch.ndata['h']
        
        # perform inference
        preds = self.model(graph_batch, feats_batch).flatten().tolist()
        
        return preds


class RDKitOracle(Oracle):

    def __init__(
        self,
        name: str,
    ):
        super().__init__()
        self.name = name

        # load descriptor
        self.descriptor = getattr(Descriptors, self.name)

    def predict(self, molecules: MoleculeCollection):
        return [self.descriptor(m.mol) for m in molecules]


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
        path: Optional[str] = None
    ):
        if not path:
            
            import tempfile
        
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:

                # prepare ligands
                self.prepare_ligands(molecules, temp_dir)
                
                # prepare command
                command = self.prepare_command(self.config, temp_dir)

                # run docking
                resp = self.dock(command)
                
                # gather results
                affinities = self.gather_results(temp_dir)

        return affinities

    def dock(self, command: str):
        import subprocess
        return subprocess.run(
            command,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
            encoding='utf-8'
        )

    def gather_results(self, directory: str):
        
        import re
        import glob
        from itertools import islice

        affinities = []
        paths = glob.glob(f'{directory}/*_out.pdbqt')

        for idx, path in enumerate(paths):
            with open(path, 'r') as file:

                # extract SMILES from the file
                smiles_str = list(islice(file, 7))[-1]
                smiles = smiles_str.split(' ')[-1].split('\n')[0]
                file.seek(0)

                # extract affinities
                affinity_strs = [line for line in file if line.startswith('REMARK VINA RESULT')]
                process_affinity = lambda s: re.search(r'-?\d+\.\d+', s).group()
                energies = [float(process_affinity(a)) for a in affinity_strs]

                # compute boltzmann sum
                affinity = self.boltzmann_sum(energies)

                # append to affinities
                affinities.append(affinity)

        return affinities

    def prepare_command(self, config, directory: str):
        
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

    def prepare_ligands(self, molecules, directory: str):
        
        import os
        
        paths = []
        for mol in molecules:
            
            # compute PDBQT
            pdbqt = self.get_pdbqt(mol.mol)
            path = os.path.join(
                directory,
                f'{mol.name}.pdbqt'
            )

            # write PDBQTs to disk
            with open(path, 'w') as file:
                file.write(pdbqt)
            paths.append(path)

        # write ligand text file
        ligands_txt = ' '.join(paths)
        path = os.path.join(directory, 'ligands.txt')
        with open(path, 'w') as file:
            file.write(ligands_txt)

    def get_pdbqt(self, mol):

        # add hydrogens (without regard to pH)
        protonated_mol = rdkit.Chem.AddHs(mol)

        # generate 3D coordinates for the ligand. 
        rdkit.Chem.AllChem.EmbedMolecule(protonated_mol)

        # initialize preparation
        preparator = MoleculePreparation(rigid_macrocycles=True)
        setup = preparator.prepare(protonated_mol)[0]
        
        # print PDBQT
        pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)
        
        return pdbqt_string

    def boltzmann_sum(self, energies, temperature=298.15):

        # Boltzmann constant in kcal/(mol·K) multiplied by temperature in K
        kT = 0.0019872041 * temperature
        
        # Energies should be in kcal/mol
        energies = np.array(energies)

        # Use logsumexp for numerical stability
        boltz_sum = -kT * np.log(np.sum(np.exp(-energies / kT)))
        
        return boltz_sum


class NeuralOracle(Oracle):

    def __init__(
        self,
        name: str
    ):
        super().__init__()
        self.name = name

        # load 


def build_model_2d(config=None):
    """
    Build appropriate 2D graph model.

    Parameters
    ----------
    config : Union[str, dict], optional
        Either a dict or JSON file with model config options. If not passed,
        `config` will be taken from `wandb`.

    Returns
    -------
    mtenn.conversion_utils.GAT
        GAT graph model
    """
    from dgllife.utils import CanonicalAtomFeaturizer
    from mtenn.conversion_utils import GAT

    # config.update({"in_node_feats": CanonicalAtomFeaturizer().feat_size()})
    in_node_feats = CanonicalAtomFeaturizer().feat_size()

    model = GAT(
        in_feats=in_node_feats,
        hidden_feats=[config["gnn_hidden_feats"]] * config["num_gnn_layers"],
        num_heads=[config["num_heads"]] * config["num_gnn_layers"],
        feat_drops=[config["dropout"]] * config["num_gnn_layers"],
        attn_drops=[config["dropout"]] * config["num_gnn_layers"],
        alphas=[config["alpha"]] * config["num_gnn_layers"],
        residuals=[config["residual"]] * config["num_gnn_layers"],
    )

    return model
