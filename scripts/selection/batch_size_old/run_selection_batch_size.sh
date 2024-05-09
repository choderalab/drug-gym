#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./selection_temperature.py"

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

# Define specific batch sizes
BATCH_SIZES=(1 8 12 16 24 48 96 192 384)

# Number of trials to run for each noise level
NUM_TRIALS=200

# Run multiple trials for this noise level
for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do

    # Iterate over each batch size
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        echo "Running trial $TRIAL for batch size: $BATCH_SIZE"        

        # Submit the job with bsub
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 5:00 \
             -o "${LOGS_DIR}/batch_size_${BATCH_SIZE}_trial_${TRIAL}.stdout" \
             -eo "${LOGS_DIR}/batch_size_${BATCH_SIZE}_trial_${TRIAL}.stderr" \
             python3 "$PYTHON_SCRIPT" --batch_size "$TEMP" --out_dir "$RUN_DIR"

    done
done
echo "All trials completed."