import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

import os
os.environ['CHEMFP_LICENSE'] = (
    '20231114-columbia.edu@DAAABLGMDNEEHFALIFOLIONPFHFDJDOLHABF'
)
import chemfp

def fingerprints(path):
    return chemfp.load_fingerprints(path)

def enamine(path):
    return Chem.SDMolSupplier(path)

def dsip(path):
    """
    Load DSi-Poised Library.
    From Enamine: https://enamine.net/compound-libraries/fragment-libraries/dsi-poised-library
 
    """
    # import and process dsip
    dsip = PandasTools.LoadSDF(path)

    # filter reactions that lack reagents
    dsip = dsip[pd.notnull(dsip['reagsmi1'])].reset_index(drop=True)

    # convert smiles to rdMol
    dsip[['reagmol1', 'reagmol2']] = dsip[['reagsmi1', 'reagsmi2']].applymap(Chem.MolFromSmiles)

    # consolidate reagents
    dsip['reagents'] = [
        [
            {'id': r1, 'smiles': rs1, 'rdMol': rm1},
            {'id': r2, 'smiles': rs2, 'rdMol': rm2}
        ]
        for _, r1, r2, rs1, rm1, rs2, rm2 in
        dsip[['reag1', 'reag2', 'reagsmi1', 'reagmol1', 'reagsmi2', 'reagmol2']].itertuples()
    ]

    # subset columns
    dsip = (
        dsip[['Catalog ID', 'reaction', 'ROMol', 'reagents']]
        .rename(columns={'ROMol': 'rdMol', 'Catalog ID': 'id'})
    )
    
    return dsip


def reactions(path):
    """
    Load reactions.
    From SmilesClickChem: https://zenodo.org/record/4100676
    
    """

    # load from JSON
    reactions = pd.read_json(path).T.reset_index(drop=True)
    reactions['rdReaction'] = [AllChem.ReactionFromSmarts(smirks)
                               for smirks in reactions['reaction_string']]
    
    # process columns
    reactions['reactant_smarts'] = reactions['reaction_string'].apply(
        lambda s: s.split('>>')[0].split('.')
    )
    reactions['product_smarts'] = reactions['reaction_string'].apply(
        lambda s: s.split('>>')[-1]
    )
    reactions['reactant_rdMol'] = reactions['reactant_smarts'].apply(
        lambda x: [Chem.MolFromSmarts(s) for s in x]
    )
    reactions['product_rdMol'] = reactions['rdReaction'].apply(
        lambda x: x.GetProducts()[0]
    )
    
    # make molecule dicts
    reactions['reactants'] = [
        [{'id': ids[i], 'smarts': smarts[i], 'rdMol': rdMols[i]}
         for i in range(len(ids))]
        for _, ids, smarts, rdMols in
        reactions[['functional_groups', 'reactant_smarts', 'reactant_rdMol']].itertuples()
    ]

    reactions['product'] = [
        {'smarts': product_string, 'rdMol': product_rdMol}
        for _, product_string, product_rdMol in
        reactions[['product_smarts', 'product_rdMol']].itertuples()
    ]
    
    # subset and rename columns
    reactions = (
        reactions[['reaction_name', 'rdReaction', 'product', 'reactants']]
        .rename(columns={'reaction_name': 'reaction'})
    )
    
    return reactions
