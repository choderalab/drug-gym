#!/bin/bash

# Disable core dumps
ulimit -c 0

# Define the path to your Python script
PYTHON_SCRIPT="./selection_max_noise.py"

# Generate a timestamp for the current time
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Define the base output directory for results
BASE_OUT_DIR="/data/chodera/retchinm/max_noise"

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
    echo "Trial $TRIAL for maximum error"

    # Submit a bsub job to run the script in parallel instances
    bsub -q gpuqueue -n 4 -gpu "num=1:mig=1/1:aff=no" \
            -J drug-gym_max_error_${TRIAL} -R "rusage[mem=8G] span[hosts=1]" -W 5:59 \
            -o "${LOGS_DIR}/max_error_trial_${TRIAL}.stdout" \
            -eo "${LOGS_DIR}/max_error_trial_${TRIAL}.stderr" \
            /usr/bin/sh /home/retchinm/chodera/drug-gym/scripts/selection/max_noise/run_trial.sh $RUN_DIR
done
echo "All trials completed."