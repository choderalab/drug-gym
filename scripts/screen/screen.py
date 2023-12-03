import argparse
from rdkit import Chem
from itertools import islice
from dgym.envs.oracle import DockingOracle

# Function definitions
def batched(iterable, batch_size):

    if batch_size < 1:
        raise ValueError('`batch_size` must be >= 1')
    
    # Gather batch
    it = iter(iterable)
    while (batch := list(islice(it, batch_size))):
        batch_processed = dg.collection.MoleculeCollection([
            dg.molecule.Molecule(mol)
            for mol in batch if mol
        ])
        yield batch_processed
        
# Function to process batches and get results
def get_docking_results(batches):
        
    # get affinity
    batch_results = docking_oracle(batch)

    # attach smiles
    smiles = [m.smiles for m in batch]
    smiles_results = list(zip(smiles, batch_results))

    # append to results
    return smiles_results

# create docking oracle
config = {
    'center_x': 9.812,
    'center_y': -0.257,
    'center_z': 20.8485,
    'size_x': 14.328,
    'size_y': 8.85,
    'size_z': 12.539,
    'exhaustiveness': 128,
    'max_step': 20,
    'num_modes': 9,
    'scoring': 'vinardo',
    'refine_step': 3,
    'seed': 5
}

docking_oracle = DockingOracle(
    'Mpro affinity',
    receptor_path=f'./Mpro_prepped.pdbqt',
    config=config
)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--machine_id", type=int, help="ID of the current machine (0-indexed)")
parser.add_argument("--num_machines", type=int, help="Total number of machines")
args = parser.parse_args()

# Load all molecules to get total count
supplier = Chem.SmilesMolSupplier('./strict_fragments.cxsmiles', delimiter='\t')
total_molecules = len(supplier)

# Calculate data portion per machine
data_portion = total_molecules // args.num_machines

# Calculate start and end index for each machine
start_index = args.machine_id * data_portion
if args.machine_id == args.num_machines - 1:  # If it's the last machine
    end_index = total_molecules  # Process till the end
else:
    end_index = start_index + data_portion

# Batch size remains 300
batch_size = 300

# Get subset of molecules for this machine
supplier_subset = islice(supplier, start_index, end_index)

# Create batches and process
batches = batched(supplier_subset, batch_size)

# Check if file already exists
file_path = f'./out/strict_fragments_machine_{args.machine_id}.tsv'
header = not os.path.exists(file_path)
for batch in batches:
    results = get_docking_results(batches)
    results_df = pd.DataFrame(results, columns=['smiles', 'affinity'])
    results_df.to_csv(
        file_path,
        mode='a',
        header=header,
        index=False
    )
