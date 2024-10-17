import argparse
import dgym as dg
import torch
import pyarrow.parquet as pq
from dgym.envs.designer import Designer, Generator
# select first molecule
import random
import os
from dgym.envs.oracle import DockingOracle
import pandas as pd
from dgym.molecule import Molecule
from dgym.envs.designer import Designer, Generator
import os


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

    building_blocks = dg.datasets.disk_loader(
        f'{path}/Enamine_Building_Blocks_Stock_262336cmpd_20230630.sdf')
    fingerprints = dg.datasets.fingerprints(
        f'{path}/Enamine_Building_Blocks_Stock_262336cmpd_20230630_atoms.fpb')

    table = pq.read_table(f'{path}/sizes.parquet')[0]
    sizes = torch.tensor(table.to_numpy())

    return deck, reactions, building_blocks, fingerprints, sizes


def get_molecules(
        deck,
        reactions,
        building_blocks,
        fingerprints,
        sizes,
    ):
        
    designer = Designer(
        Generator(building_blocks, fingerprints, sizes),
        reactions,
        cache = True
    )

    
    def select_molecule(deck):
        initial_index = random.randint(0, len(deck))
        initial_molecule = deck[initial_index]
        if len(initial_molecule.reactants) == 2 \
            and designer.match_reactions(initial_molecule):
            return initial_molecule
        else:
            return select_molecule(deck)

    molecules = []
    for _ in range(20):
        
        # pick a molecule randomly from the deck
        initial_molecule = select_molecule(deck)

        # generate a few rounds of random molecules in REAL Space
        molecule = initial_molecule
        designer.reset_cache()
        for _ in range(3):
            molecule = designer.design(molecule, 1, temperature=1.0)[0]
        
        # generate a bunch of analogs
        analogs = designer.design(molecule, 15, temperature=0.0)
        molecules.extend(analogs)

    return molecules

def get_docking_config(path, target_index, scorer):
    

    dockstring_dir = f'{path}/dockstring_targets/'
    files = os.listdir(dockstring_dir)
    configs = sorted([f for f in files if 'conf' in f])
    targets = sorted([f for f in files if 'target' in f])

    with open(dockstring_dir + configs[target_index], 'r') as f:
        config_ = f.readlines()
        config_ = [c.replace('\n', '') for c in config_]
        config_ = [c.split(' = ') for c in config_ if c]
        config_ = {c[0]: float(c[1]) for c in config_}

    target_file = targets[target_index]
    target = target_file.split('_')[0]
    
    name = f'{target} affinity'
    receptor_path = f'{path}/dockstring_targets/{target_file}'
    config = {
        'search_mode': 'balanced',
        'scoring': scorer,
        'seed': 5,
        'size_x': 15.0,
        'size_y': 15.0,
        'size_z': 15.0,
        **config_
    }
    
    return name, receptor_path, config

def get_oracle(path: str, target_index: int, scorer: str):
    
    
    # Create noiseless evaluators
    name, receptor_path, config = get_docking_config(path, target_index, scorer=scorer)
    docking_oracle = DockingOracle(name, receptor_path=receptor_path, config=config)
    return docking_oracle


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--target_index", type=int, help="The index of the docking target")
parser.add_argument("--out_dir", type=str, help="Where to put the resulting JSONs")
parser.add_argument("--scorer", type=str, help="Which mode for scoring. `vina`, `vinardo`, or `gnina`.")

args = parser.parse_args()

# Run experiment

# Load all data
path = '../../../../dgym-data'
(
    deck,
    reactions,
    building_blocks,
    fingerprints,
    sizes
) = get_data(path)

# Get random molecules
molecules = get_molecules(
    deck,
    reactions,
    building_blocks,
    fingerprints,
    sizes,
)

# Get oracle
docking_oracle = get_oracle(
    path=path, target_index=args.target_index, scorer=args.scorer)

# Score molecules
scores = docking_oracle(molecules)

# Attach smiles
smiles = [m.smiles for m in molecules]
smiles_scores = list(zip(smiles, scores))
results_df = pd.DataFrame(smiles_scores, columns=['smiles', 'affinity'])

results_df['target'] = docking_oracle.name
results_df['scorer'] = args.scorer

# Write to disk

file_path = f'{args.out_dir}/screen_targets_{args.target_index}_{args.scorer}.tsv'
results_df.to_csv(
    file_path,
    mode = 'a',
    header = not os.path.exists(file_path),
    index = False,
    sep='\t',
)