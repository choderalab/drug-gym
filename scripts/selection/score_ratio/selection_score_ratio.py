import uuid
import argparse
import dgym as dg

import torch
import pyarrow.parquet as pq

import random

import os

from dgym.envs.oracle import \
    DockingOracle, CatBoostOracle, RDKitOracle, NoisyOracle
from dgym.envs.utility import ClassicUtilityFunction

from dgym.envs.utility import (
    ClassicUtilityFunction, MultipleUtilityFunction
)

from copy import deepcopy

from dgym.envs.designer import Designer, Generator

import json

from dgym.envs import DrugEnv

from dgym.agents import SequentialDrugAgent
from dgym.agents.exploration import EpsilonGreedy

from dgym.experiment import Experiment

import json
import uuid
from utils import serialize_with_class_names

def get_data(path):

    deck = dg.MoleculeCollection.load(
        f'{path}/DSi-Poised_Library_annotated.sdf',
        reactant_names=['reagsmi1', 'reagsmi2', 'reagsmi3']
    )

    reactions = dg.ReactionCollection.from_json(
        path = f'{path}/All_Rxns_rxn_library_sorted.json',
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

def get_initial_library(deck, designer):
    
    # select first molecule
    def _select_molecule(deck):
        initial_index = random.randint(0, len(deck) - 1)
        initial_molecule = deck[initial_index]
        if len(initial_molecule.reactants) == 2 \
            and designer.match_reactions(initial_molecule):
            return initial_molecule
        else:
            return _select_molecule(deck)

    initial_molecules = [_select_molecule(deck) for _ in range(5)]
    library = dg.MoleculeCollection(initial_molecules).update_annotations()
    
    return library
    
def get_docking_config(path: str, target_index: int):
    

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
    
    config = {
        'search_mode': 'detailed',
        'scoring': 'vina',
        'seed': 5,
        'size_x': 22.5,
        'size_y': 22.5,
        'size_z': 22.5,
        **config_
    }
    
    return target, config

def get_oracles(path: str, target_index: int):

    
    target, config = get_docking_config(path, target_index)

    pIC50_oracle = DockingOracle(
        f'{target} pIC50',
        receptor_path=f'{path}/dockstring_targets/{target}_target.pdbqt',
        config=config
    )
    log_P_oracle = RDKitOracle('Log P', descriptor='MolLogP')
    log_S_oracle = CatBoostOracle(
        'Log S', path='../../../dgym/envs/models/aqsolcb.model')
    
    return pIC50_oracle, log_P_oracle, log_S_oracle

def get_multiple_utility_functions(
    pIC50_oracle,
    log_P_oracle,
    log_S_oracle,
    sigma=1.0
):

    # Define utility functions
    pIC50_utility = ClassicUtilityFunction(
        pIC50_oracle, ideal=(9, 13), acceptable=(8, 13))
    log_P_utility = ClassicUtilityFunction(
        log_P_oracle, ideal=(0.5, 1.85), acceptable=(-0.5, 3.5))
    log_S_utility = ClassicUtilityFunction(
        log_S_oracle, ideal=(-3, 1), acceptable=(-4, 1))

    # Assemble assays and surrogate models
    assays = [
        pIC50_oracle,
        log_P_oracle,
        log_S_oracle,
        pIC50_oracle.surrogate(sigma=sigma),
        log_P_oracle.surrogate(sigma=sigma),
        log_S_oracle.surrogate(sigma=sigma),
    ]

    # Environment tolerates acceptable ADMET
    utility_agent = MultipleUtilityFunction(
        utility_functions = [pIC50_utility, log_P_utility, log_S_utility],
        weights = [0.8, 0.1, 0.1]
    )
    utility_env = deepcopy(utility_agent)
    utility_env.utility_functions[1].ideal = utility_env.utility_functions[1].acceptable
    utility_env.utility_functions[2].ideal = utility_env.utility_functions[2].acceptable
    
    return assays, utility_agent, utility_env

def get_agent_sequence(batch_size: int = 24, score_ratio: int = 5):
    """
    Make the sequence for the DrugAgent.
    """
    design_size_1 = max(1, score_ratio // 2)
    design_size_2 = max(1, score_ratio - design_size_1)
    design_replace = {
        'name': 'design',
        'batch_size': batch_size,
        'parameters': {'strategy': 'replace', 'size': design_size_1, 'temperature': 0.16, 'limit': 1}
    }
    design_grow = {
        'name': 'design',
        'batch_size': batch_size,
        'parameters': {'strategy': 'grow', 'size': design_size_2}
    }
    score = {
        'name': ['Noisy ABL1 pIC50', 'Noisy Log S', 'Noisy Log P'],
        'batch_size': batch_size * score_ratio,
        'parameters': {'parallel': False, 'batch_size': 40}
    }
    make = {'name': 'make', 'batch_size': batch_size}
    test = {'name': ['ABL1 pIC50', 'Log S', 'Log P'], 'batch_size': batch_size}

    return [design_replace, score, design_grow, score, make, test]

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir", type=str, help="Where to put the resulting JSONs")
parser.add_argument(
    "--batch_size", type=int, default=8, help="Batch size in each design cycle.")
parser.add_argument(
    "--score_ratio", type=int, default=5, help="Number to score for every compound tested.")
parser.add_argument(
    "--experiment_state_path", type=str, help="Path to file for loading experiment state. Overrides the other args.")
args = parser.parse_args()

# Run experiment
path = '/data/chodera/retchinm/dgym-data'

# Load all data
(
    deck,
    reactions,
    building_blocks,
    fingerprints,
    sizes
) = get_data(path)

print('Loaded data.', flush=True)

# Get starting library
designer = Designer(
    Generator(building_blocks, fingerprints, sizes), reactions, cache = True)
library = get_initial_library(deck, designer)

print('Loaded library and designer.', flush=True)

# Get Oracles
(
    pIC50_oracle,
    log_P_oracle,
    log_S_oracle
) = get_oracles(
    path=path,
    target_index=0
)

print('Loaded oracles.', flush=True)

# Load experiment state off disk if available
try:
    with open(args.experiment_state_path, 'r') as f:
        experiment_state = json.load(f)
        args_dict = vars(args)
        for key in ['batch_size', 'score_ratio']:
            args_dict[key] = experiment_state[key]
except:
    experiment_state = {}

# Create multiple utility functions
(
    assays,
    utility_agent,
    utility_env
) = get_multiple_utility_functions(
    pIC50_oracle,
    log_P_oracle,
    log_S_oracle,
)

print('Loaded utility functions.', flush=True)

# Create DrugEnv
drug_env = DrugEnv(
    designer = designer,
    library = library,
    assays = assays,
    utility_function = utility_env
)

print('Loaded DrugEnv.', flush=True)

# Create DrugAgent
sequence = get_agent_sequence(batch_size=args.batch_size, score_ratio=args.score_ratio)
drug_agent = SequentialDrugAgent(
    sequence = sequence,
    exploration_strategy = EpsilonGreedy(epsilon=0.2),
    utility_function = utility_agent
)
print('Loaded DrugAgent.', flush=True)

# Create and run Experiment
experiment = Experiment(drug_agent=drug_agent, drug_env=drug_env).load(experiment_state)
file_path = args.experiment_state_path \
    or f'{args.out_dir}/selection_batch_size_{args.batch_size}_score_ratio_{args.score_ratio}_{uuid.uuid4()}.json'
result = experiment.run(**vars(args), out=file_path)[0]

# Export results

result_serialized = serialize_with_class_names(result)
with open(file_path, 'w') as f:
    json.dump(result_serialized, f)