import os
import dgym as dg
import pandas as pd
import argparse
from rdkit import Chem
from itertools import islice
from dgym.envs.oracle import DockingOracle
import random

import dgym as dg
import torch
import pyarrow.parquet as pq
# Docking oracles
from dgym.envs.oracle import DockingOracle, NoisyOracle
from dgym.envs.utility import ClassicUtilityFunction
from dgym.envs.designer import Designer, Generator
import uuid



def get_data(path):

    deck = dg.MoleculeCollection.load(
        f'{path}/DSi-Poised_Library_annotated.sdf',
        reactant_names=['reagsmi1', 'reagsmi2', 'reagsmi3']
    )

    reactions = dg.ReactionCollection.from_json(
        path = f'{path}/All_Rxns_rxn_library.json',
        smarts_col = 'reaction_string',
        classes_col = 'functional_groups'
    )

    building_blocks = dg.datasets.disk_loader(f'{path}/Enamine_Building_Blocks_Stock_262336cmpd_20230630.sdf')
    fingerprints = dg.datasets.fingerprints(f'{path}/Enamine_Building_Blocks_Stock_262336cmpd_20230630_atoms.fpb')

    
    table = pq.read_table(f'{path}/sizes.parquet')[0]
    sizes = torch.tensor(table.to_numpy())

    return deck, reactions, building_blocks, fingerprints, sizes

def get_oracles(path, sigma=0.1):

    config = {
        'center_x': 44.294,
        'center_y': 28.123,
        'center_z': 2.617,
        'size_x': 22.5,
        'size_y': 22.5,
        'size_z': 22.5,
        'search_mode': 'balanced',
        'scoring': 'gnina',
        'seed': 5
    }

    # Create noiseless evaluators
    docking_oracle = DockingOracle(
        'ADAM17 affinity',
        receptor_path=f'{path}/ADAM17.pdbqt',
        config=config
    )

    return docking_oracle

# Function to process batches and get results
def get_docking_results(batches):
        
    # get affinity
    batch_results = docking_oracle(batch)

    # attach smiles
    smiles = [m.smiles for m in batch]
    smiles_results = list(zip(smiles, batch_results))

    # append to results
    return smiles_results

def select_molecule(deck):
    initial_index = random.randint(0, len(deck))
    initial_molecule = deck[initial_index]
    if len(initial_molecule.reactants) == 2 \
        and designer.match_reactions(initial_molecule):
        return initial_molecule
    else:
        return select_molecule(deck)

def get_molecules(deck, batch_size):

    random_molecules = []
    for _ in range(batch_size // 10):

        # pick a molecule randomly from the deck
        initial_molecule = select_molecule(deck)

        # generate a few rounds of random molecules in REAL Space
        molecule = initial_molecule
        designer.reset_cache()
        for _ in range(3):
            molecule = designer.design(molecule, 1, temperature=1.0)[0]

        # generate a bunch of analogs
        molecules = designer.design(molecule, batch_size // 30, temperature=1.0)

        # add molecules to random molecules
        random_molecules.extend(molecules)
    
    return random_molecules



# load all data
path = '../../../../dgym-data'

(
    deck,
    reactions,
    building_blocks,
    fingerprints,
    sizes
) = get_data(path)


designer = Designer(
    Generator(building_blocks, fingerprints, sizes),
    reactions,
    cache = True
)

# create docking oracle
config = {
    'center_x': 44.294,
    'center_y': 28.123,
    'center_z': 2.617,
    'size_x': 22.5,
    'size_y': 22.5,
    'size_z': 22.5,
    'search_mode': 'balanced',
    'scoring': 'gnina',
    'seed': 5
}

docking_oracle = DockingOracle(
    'ADAM17 affinity',
    receptor_path=f'./ADAM17.pdbqt',
    config=config
)

# Batch size is 300
batch_size = 300

# Check if file already exists

file_path = f'./out/adam17_random_batch_{uuid.uuid4()}.tsv'
for _ in range(10):
    
    # Get subset of molecules for this machine
    batch = get_molecules(deck, batch_size)

    # Get docking results
    results = get_docking_results(batch)
    
    # Write to disk
    results_df = pd.DataFrame(results, columns=['smiles', 'affinity'])
    results_df.to_csv(
        file_path,
        mode = 'a',
        header = not os.path.exists(file_path),
        index = False,
        sep='\t',
    )
