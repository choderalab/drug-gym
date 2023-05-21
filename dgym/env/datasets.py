import pandas as pd
from rdkit import Chem
from typing import Optional
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from drug_gym.reaction import Reaction

import os
os.environ['CHEMFP_LICENSE'] = (
    '20231114-columbia.edu@DAAABLGMDNEEHFALIFOLIONPFHFDJDOLHABF'
)
import chemfp

def fingerprints(path):
    return chemfp.load_fingerprints(path)

def enamine(path):
    return Chem.SDMolSupplier(path)
