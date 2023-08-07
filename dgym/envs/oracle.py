import dgl
import torch
import dgllife
from rdkit.Chem import Descriptors
from dgym.collection import Collection, MoleculeCollection

class Oracle:
    
    def __init__(self) -> None:
        self.cache = {}

    def __call__(self, molecules: Collection):
        
        # identify uncached molecules
        in_cache = lambda m: m.smiles not in self.cache
        
        if uncached_molecules := molecules.filter(in_cache):
            
            # make predictions
            preds = self.predict(uncached_molecules)
            
            # cache results
            self.cache.update(zip(uncached_molecules.smiles, preds))

        # fetch all results (old and new) from cache
        return [self.cache[m.smiles] for m in molecules]

    def predict(self, molecules: Collection):
        raise NotImplementedError


class DGLOracle(Oracle):

    def __init__(
        self,
        model_name: str,
        mol_to_graph=dgllife.utils.MolToBigraph(
            add_self_loop=True,
            node_featurizer=dgllife.utils.CanonicalAtomFeaturizer()
        )
    ):
        super().__init__()
        self.model_name = model_name
        self.mol_to_graph = mol_to_graph

        # load model
        self.model = dgllife.model.load_pretrained(model_name, log=False)
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
        descriptor_name: str,
    ):
        super().__init__()
        self.descriptor_name = descriptor_name

        # load descriptor
        self.descriptor = getattr(Descriptors, self.descriptor_name)

    def predict(self, molecules: MoleculeCollection):
        return [self.descriptor(m.mol) for m in molecules]