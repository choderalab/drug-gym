import chemfp
import pandas as pd
from rdkit import Chem
from typing import Optional
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from dgym.reaction import Reaction

chemfp.set_license('20241121-columbia.edu@DAAAPLPPDDKGPECJIJJGFNBEPIIKHOOMFAOG')

def fingerprints(path):
    return chemfp.load_fingerprints(path)

def disk_loader(path, format='sdf'):
    if format == 'sdf':
        supplier = Chem.SDMolSupplier(path)
    elif format == 'smiles':
        supplier = Chem.MultithreadedSmilesMolSupplier(path)
    return supplier