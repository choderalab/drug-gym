import pandas as pd
from rdkit import Chem
from typing import Optional
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from dgym.reaction import Reaction

import os
os.environ['CHEMFP_LICENSE'] = (
    '20231114-columbia.edu@DAAABLGMDNEEHFALIFOLIONPFHFDJDOLHABF'
)
import chemfp

def fingerprints(path):
    return chemfp.load_fingerprints(path)

def enamine(path):
    return Chem.SDMolSupplier(path)


from dgllife.utils.mol_to_graph import ToGraph, MolToBigraph
class DGLLifeUnlabeledDataset(object):
    """Construct a SMILES dataset without labels for inference.

    We will 1) Filter out invalid SMILES strings and record canonical SMILES strings
    for valid ones 2) Construct a DGLGraph for each valid one and feature its node/edge

    Parameters
    ----------
    collection : dgym.Collection
        Contains the molecules of interest.
    mol_to_graph: callable, rdkit.Chem.rdchem.Mol -> DGLGraph
        A function turning an RDKit molecule object into a DGLGraph.
        Default to :func:`dgllife.utils.mol_to_bigraph`.
    node_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    """
    def __init__(self,
                 collection,
                 mol_to_graph=None,
                 node_featurizer=None,
                 edge_featurizer=None,
                 log_every=1000):
        super(DGLLifeUnlabeledDataset, self).__init__()

        self.smiles = collection.smiles
        self.graphs = []
        
        mol_list = []
        for m in collection.molecules:
            m.mol.UpdatePropertyCache()
            mol_list.append(m.mol)
        
        if mol_to_graph is None:
            mol_to_graph = MolToBigraph()

        # Check for backward compatibility
        if isinstance(mol_to_graph, ToGraph):
            assert node_featurizer is None, \
                'Initialize mol_to_graph object with node_featurizer=node_featurizer'
            assert edge_featurizer is None, \
                'Initialize mol_to_graph object with edge_featurizer=edge_featurizer'
        else:
            mol_to_graph = partial(mol_to_graph, node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer)

        for i, mol in enumerate(mol_list):
            if (i + 1) % log_every == 0:
                print('Processing molecule {:d}/{:d}'.format(i + 1, len(self)))
            self.graphs.append(mol_to_graph(mol))

    def __getitem__(self, item):
            """Get datapoint with index

            Parameters
            ----------
            item : int
                Datapoint index

            Returns
            -------
            str
                SMILES for the ith datapoint
            DGLGraph
                DGLGraph for the ith datapoint
            """
            return self.smiles[item], self.graphs[item]


    def __len__(self):
            """Size for the dataset

            Returns
            -------
            int
                Size for the dataset
            """
            return len(self.graphs)