import uuid
import argparse
import dgym as dg
import pandas as pd

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

def get_agent_sequence(temperature: float):
    """
    Make the sequence for the DrugAgent.
    """

    design_grow = {'name': 'design', 'batch_size': 24, 'parameters': {'strategy': 'grow', 'size': 5}}
    design_replace = {
        'name': 'design',
        'batch_size': 24,
        'parameters': {'strategy': 'replace', 'size': 5, 'temperature': temperature, 'limit': 10}
    }
    score = {'name': ['Noisy ABL1 pIC50', 'Noisy Log S', 'Noisy Log P'], 'batch_size': 24 * 5}
    make = {'name': 'make', 'batch_size': 24}
    test = {'name': ['ABL1 pIC50', 'Log S', 'Log P'], 'batch_size': 24} # 8
    design_and_score = [design_replace, score]

    return [*(design_and_score * 1), design_grow, score, make, test]

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir", type=str, help="Where to put the resulting JSONs")
parser.add_argument(
    "--temperature", type=float, help="Boltzmann temperature for building block sampling")
args = parser.parse_args()

# Run experiment
path = '../../../../dgym-data'

# Load all data
(
    deck,
    reactions,
    building_blocks,
    fingerprints,
    sizes
) = get_data(path)

# Get starting library

designer = Designer(
    Generator(building_blocks, fingerprints, sizes), reactions, cache = True)
library = get_initial_library(deck, designer)

# Get Oracles
(
    pIC50_oracle,
    log_P_oracle,
    log_S_oracle
) = get_oracles(
    path=path,
    target_index=0
)

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

# Create DrugEnv

drug_env = DrugEnv(
    designer = designer,
    library = library,
    assays = assays,
    utility_function = utility_env
)

# Create DrugAgent

sequence = get_agent_sequence(temperature = args.temperature)
drug_agent = SequentialDrugAgent(
    sequence = sequence,
    exploration_strategy = EpsilonGreedy(epsilon=0.2),
    utility_function = utility_agent
)

# Create and run Experiment

experiment = Experiment(
    drug_agent=drug_agent, drug_env=drug_env)
file_path = f'{args.out_dir}/selection_temperature_{args.temperature}_{uuid.uuid4()}.json'
result = experiment.run(**vars(args), out=file_path)[0]

# Export results


result_serialized = serialize_with_class_names(result)
json.dump(result_serialized, open(file_path, 'w'))