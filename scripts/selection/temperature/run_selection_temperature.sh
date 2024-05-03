#!/bin/bash

# Disable core dumps
ulimit -c 0

# Define the path to your Python script
PYTHON_SCRIPT="./selection_temperature.py"

# Generate a timestamp for the current time
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the base output directory for results
BASE_OUT_DIR="/data/chodera/retchinm/temperature"

# Create a new directory for this run with the timestamp
RUN_DIR="${BASE_OUT_DIR}/${TIMESTAMP}"
LOGS_DIR="${RUN_DIR}/logs"

# Make sure the new directories exist
mkdir -p "$RUN_DIR"
mkdir -p "$LOGS_DIR"

# Number of trials to run for each temperature level
NUM_TRIALS=25

# Number of parallel processes within each job
NUM_PARALLEL=4

# Run multiple trials for each temperature level
for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
    # Run the python script several times in parallel within a single bsub job for each temperature from 0 to 20
    for TEMP in $(seq 0 20); do
        echo "Trial $TRIAL for temperature $TEMP"

        # Submit a bsub job to run the script in parallel instances
        bsub -q gpuqueue -n 4 -gpu "num=1" -R "rusage[mem=8] span[hosts=1]" -W 5:59 \
             -o "${LOGS_DIR}/temp_${TEMP}_trial_${TRIAL}.stdout" \
             -eo "${LOGS_DIR}/temp_${TEMP}_trial_${TRIAL}.stderr" \
             "for i in $(seq -s ' ' 1 $NUM_PARALLEL); do python3 '$PYTHON_SCRIPT' --temperature $((TEMP)) --out_dir '$RUN_DIR' & done; wait"
    done
    echo "Completed all trials for temperature level: $TEMP"
done
echo "All trials completed."
