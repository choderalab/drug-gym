#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./selection_noise.py"

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

# Define start, end, and increment for noise levels
START=0
END=20  # For noise levels up to 2
INCREMENT=1

# Number of trials to run for each noise level
NUM_TRIALS=50

# Run multiple trials for this noise level
for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do

    # Generate noise levels from 0 to 2 with a step of 0.1
    for NOISE_INT in $(seq $START $INCREMENT $END); do
        NOISE=$(echo "scale=2; $NOISE_INT / 10" | bc)
        echo "Trial $TRIAL for noise $NOISE"
        
        # Submit the job with bsub
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 0:30 \
             -o "${LOGS_DIR}/logs/noise_${NOISE}_trial_${TRIAL}.stdout" \
             -eo "${LOGS_DIR}/logs/noise_${NOISE}_trial_${TRIAL}.stderr" \
             python3 "$PYTHON_SCRIPT" --sigma "$NOISE" --out_dir "$RUN_DIR"
    done
    
    echo "Completed all trials for noise level: $NOISE"
done

echo "All trials completed."