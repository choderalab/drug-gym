# Number of machines (processors)
num_machines=40

# Outer loop to submit jobs
for machine_id in $(seq 0 $((num_machines - 1))); do

    # Submit job
    bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 5:00 \
         -o "logs_$machine_id.stdout" -eo "logs_$machine_id.stderr" \
         python3 adam17_screen.py
done
