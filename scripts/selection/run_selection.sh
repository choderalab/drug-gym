#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./selection.py"

# Define the output directory for results
OUT_DIR="./out"

# Make sure the output directory exists
mkdir -p "$OUT_DIR"

# Define start, end, and increment for noise levels
START=0
END=20  # For noise levels up to 2
INCREMENT=1

# Number of trials to run for each noise level
NUM_TRIALS=5

# Generate noise levels from 0 to 2 with a step of 0.1
for NOISE_INT in $(seq $START $END $INCREMENT); do
    NOISE=$(echo "scale=2; $NOISE_INT / 10" | bc)
    echo "Running trials for noise level: $NOISE"
    
    # Run multiple trials for this noise level
    for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
        echo "Trial $TRIAL for noise $NOISE"
        
        # Submit the job with bsub
        bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 1:00 \
             -o "$OUT_DIR/logs/noise_${NOISE}_trial_${TRIAL}.stdout" \
             -eo "$OUT_DIR/logs/noise_${NOISE}_trial_${TRIAL}.stderr" \
             python3 "$PYTHON_SCRIPT" --sigma "$NOISE" --out_dir "$OUT_DIR"
    done
    
    echo "Completed all trials for noise level: $NOISE"
done

echo "All trials completed."