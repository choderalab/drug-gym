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

# Define start, end, and increment for temperature levels
START=0
END=20
INCREMENT=1

# Number of trials to run for each level
NUM_TRIALS=50

# Generate temperature levels from START to END with a step of INCREMENT
for TEMP_INT in $(seq $START $INCREMENT $END); do
    TEMP=$(echo "scale=2; $TEMP_INT / 20" | bc)
    echo "Running trials for temperature level: $TEMP"

    # Run multiple trials for this temperature level
    for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
        echo "Trial $TRIAL for temperature $TEMP"

        # Submit the job with bsub
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 0:05 \
             -o "${LOGS_DIR}/temp_${TEMP}_trial_${TRIAL}.stdout" \
             -eo "${LOGS_DIR}/temp_${TEMP}_trial_${TRIAL}.stderr" \
             python3 "$PYTHON_SCRIPT" --temperature "$TEMP" --out_dir "$RUN_DIR"
    done

    echo "Completed all trials for temperature level: $TEMP"
done
