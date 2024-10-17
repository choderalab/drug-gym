import argparse
import dgym as dg

# Docking oracles
from dgym.envs.oracle import DockingOracle, NoisyOracle
from dgym.envs.utility import ClassicUtilityFunction

import pandas as pd
from dgym.molecule import Molecule
from dgym.envs.designer import Designer, Generator
from dgym.envs.drug_env import DrugEnv
from dgym.agents import SequentialDrugAgent
from dgym.agents.exploration import EpsilonGreedy
from dgym.experiment import Experiment

import json
import uuid
from utils import serialize_with_class_names

# load all data
path = '../../../../dgym-data'

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
fingerprints = dg.datasets.fingerprints(f'{path}/out/Enamine_Building_Blocks_Stock_262336cmpd_20230630_atoms.fpb')


config = {
    'center_x': 44.294,
    'center_y': 28.123,
    'center_z': 2.617,
    'size_x': 22.5,
    'size_y': 22.5,
    'size_z': 22.5,
    'exhaustiveness': 128,
    'max_step': 20,
    'num_modes': 9,
    'search_mode': 'fast',
    'scoring': 'gnina',
    'refine_step': 3,
    'seed': 5
}

# Create noiseless evaluators
docking_oracle = DockingOracle(
    'ADAM17 affinity',
    receptor_path=f'{path}/ADAM17.pdbqt',
    config=config
)

docking_utility = ClassicUtilityFunction(
    docking_oracle,
    ideal=(7.5, 9.5),
    acceptable=(7.125, 9.5)
)

# Create noisy evaluator
noisy_docking_oracle = NoisyOracle(
    docking_oracle,
    sigma=0.1
)

noisy_docking_utility = ClassicUtilityFunction(
    noisy_docking_oracle,
    ideal=(7.5, 9.5),
    acceptable=(7.125, 9.5)
)



designer = Designer(
    Generator(building_blocks, fingerprints),
    reactions,
    cache = True
)

initial_library = dg.MoleculeCollection([deck[659]])
initial_library.update_annotations()

drug_env = DrugEnv(
    designer,
    library = initial_library,
    assays = [docking_oracle],
    budget = 500,
    utility_function = docking_utility,
)

sequence = [
    {'name': 'ideate', 'parameters': {'temperature': 0.1, 'size': 10, 'strict': False}},
    {'name': 'ideate', 'parameters': {'temperature': 0.0, 'size': 10, 'strict': True}},
    {'name': 'ADAM17 affinity'},
]

drug_agent = SequentialDrugAgent(
    sequence = sequence,
    utility_function = noisy_docking_utility,
    exploration_strategy = EpsilonGreedy(epsilon = 0.0),
    branch_factor = 2
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, help="Spread of the noise distribution")
parser.add_argument("--out_dir", type=str, help="Where to put the resulting JSONs")

args = parser.parse_args()

drug_agent.utility_function.oracle.sigma = args.sigma
experiment = Experiment(drug_agent, drug_env)
result = experiment.run(**vars(args))

# Export results

file_path = f'{args.out_dir}/selection_noise_{uuid.uuid4()}.json'
result_serialized = serialize_with_class_names(result)
json.dump(result_serialized, open(file_path, 'w'))