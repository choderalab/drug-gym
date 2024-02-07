#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./selection_temperature.py"

# Define the output directory for results
OUT_DIR="./out"

# Make sure the output directory exists
mkdir -p "$OUT_DIR"

# Define start, end, and increment for noise levels
START=0
END=10
INCREMENT=1

# Number of trials to run for each level
NUM_TRIALS=50

# Generate noise levels from 0 to 2 with a step of 0.1
for TEMP_INT in $(seq $START $INCREMENT $END); do
    TEMP=$(echo "scale=2; $TEMP_INT / 10" | bc)
    echo "Running trials for design level: $TEMP"
    
    # Run multiple trials for this noise level
    for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
        echo "Trial $TRIAL for noise $TEMP"
        
        # Submit the job with bsub
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 1:00 \
             -o "$OUT_DIR/logs/temp_${TEMP}_trial_${TRIAL}.stdout" \
             -eo "$OUT_DIR/logs/temp_${TEMP}_trial_${TRIAL}.stderr" \
             python3 "$PYTHON_SCRIPT" --temperature "$TEMP" --out_dir "$OUT_DIR"
    done
    
    echo "Completed all trials for noise level: $NOISE"
done

echo "All trials completed."