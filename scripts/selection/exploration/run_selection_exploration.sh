#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./selection_exploration.py"

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
END=10  # For noise levels up to 2
INCREMENT=1

# Number of trials to run for each batch size
NUM_TRIALS=50

# Generate epsilon levels from 0 to 1 with a step of 0.1
# Run multiple trials for this batch size
for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
    echo "Trial $TRIAL"
    
    for EPSILON_INT in $(seq $START $INCREMENT $END); do
        EPSILON=$(echo "scale=2; $EPSILON_INT / 10" | bc)
        echo "Running $TRIAL trial for noise level: $EPSILON"

        # Submit the job with bsub
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 0:05 \
             -o "${LOGS_DIR}/epsilon_${EPSILON}_trial_${TRIAL}.stdout" \
             -eo "${LOGS_DIR}/epsilon_${EPSILON}_trial_${TRIAL}.stderr" \
             python3 "$PYTHON_SCRIPT" --epsilon "$EPSILON" --out_dir "$RUN_DIR"
    done

    echo "Completed all trials for epsilon: $EPSILON"
done
