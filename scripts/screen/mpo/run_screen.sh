#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./mpo_screen.py"

# Generate a timestamp for the current time
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the base output directory for results
BASE_OUT_DIR="./out"

# Create a new directory for this run with the timestamp
RUN_DIR="${BASE_OUT_DIR}/${TIMESTAMP}"
LOGS_DIR="${RUN_DIR}/logs"

# Make sure the new directories exist
mkdir -p "$RUN_DIR"
mkdir -p "$LOGS_DIR"

# Number of trials to run for each target
NUM_TRIALS=100

# Run multiple trials for each scorer
for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do

    echo "Trial $TRIAL"
    
    # Submit the job with bsub
    bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 0:10 \
            -o "${LOGS_DIR}/mpo_trial_${TRIAL}.stdout" \
            -eo "${LOGS_DIR}/mpo_trial_${TRIAL}.stderr" \
            python3 "$PYTHON_SCRIPT" --out_dir "$RUN_DIR"
done
echo "Completed all trials"