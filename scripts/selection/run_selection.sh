#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT="./selection.py"

# Define the output directory for results
OUT_DIR="./out"

# Make sure the output directory exists
mkdir -p "$OUT_DIR"

# Define start, end, and increment for noise levels
START=0
END=20
INCREMENT=1

# Number of trials to run for each noise level
NUM_TRIALS=5

# Generate noise levels from 0 to 1 with a step of 0.1
for NOISE_INT in $(seq $START $END $INCREMENT); do
    NOISE=$(echo "scale=2; $NOISE_INT / 10" | bc)
    echo "Running trials for noise level: $NOISE"
    
    # Run multiple trials for this noise level
    for (( TRIAL=1; TRIAL<=NUM_TRIALS; TRIAL++ )); do
        echo "Trial $TRIAL for noise $NOISE"
        
        # Call the Python script with the current noise level
        # Redirecting stdout and stderr to log files
        python3 "$PYTHON_SCRIPT" --out_dir "$OUT_DIR" --sigma "$NOISE" > "logs/noise_${NOISE}_trial_${TRIAL}.stdout" 2> "logs/noise_${NOISE}_trial_${TRIAL}.stderr"
        
        if [ $? -eq 0 ]; then
            echo "Trial $TRIAL for noise $NOISE completed successfully."
        else
            echo "Trial $TRIAL for noise $NOISE failed. Check logs for details."
        fi
    done
    
    echo "Completed all trials for noise level: $NOISE"
done

echo "All trials completed."