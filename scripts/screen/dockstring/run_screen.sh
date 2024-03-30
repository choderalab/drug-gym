#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./dockstring_screen.py"

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

# Define start, end, and increment for target index
START=0
END=58
INCREMENT=1

# Number of trials to run for each target
NUM_TRIALS=3

# Define an array of scorers
SCORERS=("vina" "vinardo" "gnina")

# Run multiple trials for each scorer
for SCORER in "${SCORERS[@]}"; do
    echo "Running trials for scorer: $SCORER"
    for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do

        # Generate noise levels from 0 to 2 with a step of 0.1
        for TARGET in $(seq $START $INCREMENT $END); do
            echo "Trial $TRIAL for target $TARGET with scorer $SCORER"
            
            # Submit the job with bsub
            bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 0:10 \
                 -o "${LOGS_DIR}/${SCORER}_target_${TARGET}_trial_${TRIAL}.stdout" \
                 -eo "${LOGS_DIR}/${SCORER}_target_${TARGET}_trial_${TRIAL}.stderr" \
                 python3 "$PYTHON_SCRIPT" --target_index "$TARGET" --out_dir "$RUN_DIR" --scorer "$SCORER"
        done
        
        echo "Completed all trials for target index: $TARGET with scorer $SCORER"
    done
done

echo "All trials for all scorers completed."
