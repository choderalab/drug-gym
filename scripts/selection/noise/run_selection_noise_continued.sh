#!/bin/bash

# Source the LSF configuration profile
source /admin/lsftest/conf/profile.lsf

# Disable core dumps
ulimit -c 0

# Define the path to your Python script
PYTHON_SCRIPT="./selection_noise.py"

# Define output directory for results
RUN_DIR="/data/chodera/retchinm/noise/2024-05-10_23-36-56"
LOGS_DIR="${RUN_DIR}/logs"

# Make sure the new directories exist
mkdir -p "$RUN_DIR"
mkdir -p "$LOGS_DIR"

# Loop over each JSON file in the directory
for EXPERIMENT_STATE in "$RUN_DIR"/*.json; do

    # Extract the filename from the path
    EXPERIMENT_STATE_FILENAME=$(basename "$EXPERIMENT_STATE")

    # Use tail to get the last 2 characters of the file and head to pick the second last character
    LAST_SECOND_CHAR=$(tail -c 2 "$EXPERIMENT_STATE" | head -c 1)

    # Check if the penultimate character of the content is '1'
    if [[ "$LAST_SECOND_CHAR" == "1" ]]; then
        echo "Skipping file: $EXPERIMENT_STATE_FILENAME terminated successfully."
        continue
    fi

    # Submit a bsub job to run the script in parallel instances
    bsub -q gpuqueue -n 4 -gpu "num=1:j_exclusive=yes:mode=shared" -R "rusage[mem=8G] span[hosts=1]" -W 5:59 \
            -o "${LOGS_DIR}/${EXPERIMENT_STATE_FILENAME}.stdout" \
            -eo "${LOGS_DIR}/${EXPERIMENT_STATE_FILENAME}.stderr" \
            python3 $PYTHON_SCRIPT --experiment_state_path "$EXPERIMENT_STATE" --out_dir "$RUN_DIR"
    echo "Submitted experiment: $EXPERIMENT_STATE"
done
echo "All trials completed."