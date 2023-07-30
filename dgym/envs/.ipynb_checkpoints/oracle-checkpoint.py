import os
import dgllife
from dgym.collection import Collection

class Oracle:
    
    def __init__(self) -> None:
        self.cache = {}

    def predict(self, molecules: Collection):
        raise NotImplementedError

class DGLOracle(Oracle):

    def __init__(
        self,
        model_name: str,
        mol_to_graph=dgllife.utils.MolToBigraph(
            add_self_loop=True,
            node_featurizer=CanonicalAtomFeaturizer()
        )
    ):
        super().__int__()
        self.model_name = model_name
        self.mol_to_graph = mol_to_graph

        # load model
        self.model = dgllife.model.load_pretrained(model_name, log=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict(self, molecules: Collection):
        
        # identify uncached molecules
        uncached_molecules = molecules.filter(lambda m: m.smiles not in self.cache)
        
        # featurize
        graphs = [self.mol_to_graph(m.update_cache().mol) for m in uncached_molecules]
        graph_batch = dgl.batch(graphs)
        feats_batch = graph_batch.ndata['h']
        
        # perform inference
        preds = self.model(graph_batch, feats_batch).flatten().tolist()

        # cache results
        self.cache.update(zip(uncached_molecules.smiles, preds))

        # fetch all results (old and new) from cache
        return [self.cache[m.smiles] for m in molecules]
