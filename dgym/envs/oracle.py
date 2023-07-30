import os
from dgym.env.datasets import DGLLifeDataset
import dgllife

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
        new = molecules.filter(lambda m: m.smiles not in self.cache)
        
        # featurize
        graphs = [self.mol_to_graph(m.update_cache().mol) for m in new]
        graph_batch = dgl.batch(new_molecule_graphs)
        feats_batch = graph_batch.ndata['h']
        
        # perform inference
        preds = self.model(graph_batch, feats_batch).flatten().tolist()

        # cache results
        self.cache.update(zip(new.smiles, results))

        # Fetch all results (old and new) from cache
        return [self.cache[mol.smiles] for molecule in molecules]


def get_oracles(properties = []):
    oracles = {}
    for prop in properties:
        model = load_pretrained(prop, log=False)
        model = model.eval()
        oracles[prop] = model
    return oracles

def get_properties(collection, properties = ['GCN_canonical_Lipophilicity']):

    def _make_dataset(collection):
        return DGLLifeUnlabeledDataset(
            collection,
            mol_to_graph=dgllife.utils.MolToBigraph(
                add_self_loop=True,
                node_featurizer=CanonicalAtomFeaturizer(),
            )
        )

    # TODO - make actual loop over properties
    for prop in properties:
        dataset = _make_dataset(collection)
        graph_batch = dgl.batch(dataset.graphs)
        feats_batch = graph_batch.ndata['h']

        oracle = oracles[prop]
        preds = oracle(graph_batch, feats_batch).flatten().tolist()

    return preds