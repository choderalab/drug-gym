#!/bin/bash

# Source the LSF configuration profile
source /admin/lsftest/conf/profile.lsf

# Disable core dumps
ulimit -c 0

# Define the path to your Python script
PYTHON_SCRIPT="./selection_noise.py"

# Generate a timestamp for the current time
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the base output directory for results
BASE_OUT_DIR="/data/chodera/retchinm/noise"

# Create a new directory for this run with the timestamp
RUN_DIR="${BASE_OUT_DIR}/${TIMESTAMP}"
LOGS_DIR="${RUN_DIR}/logs"

# Make sure the new directories exist
mkdir -p "$RUN_DIR"
mkdir -p "$LOGS_DIR"

# Number of trials to run for each noise level
NUM_TRIALS=100

# Number of parallel processes within each job
NUM_PARALLEL=1

# Run multiple trials for each noise level
for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
    # Run the python script several times in parallel for each noise from 0.0 to 2.0
    for NOISE_INT in $(seq 0 5 20); do
        NOISE=$(echo "scale=2; $NOISE_INT / 10" | bc)
        echo "Trial $TRIAL for noise $NOISE"

        # Submit a bsub job to run the script in parallel instances
        bsub -q gpuqueue -n 4 -gpu "num=1:j_exclusive=yes:mode=shared" -R "rusage[mem=8] span[hosts=1]" -W 5:59 \
             -o "${LOGS_DIR}/temp_${NOISE}_trial_${TRIAL}.stdout" \
             -eo "${LOGS_DIR}/temp_${NOISE}_trial_${TRIAL}.stderr" \
             "for i in $(seq -s ' ' 1 $NUM_PARALLEL); do python3 '$PYTHON_SCRIPT' --sigma $NOISE --out_dir '$RUN_DIR' & done; wait"
    done
    echo "Completed all trials for noise level: $NOISE"
done
echo "All trials completed."
