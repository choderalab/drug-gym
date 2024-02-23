# Calculate total number of molecules
total_molecules=$(wc -l < "./strict_fragments.cxsmiles")

# Number of machines (processors)
num_machines=40

# Data portion per machine
molecules_per_machine=$((total_molecules / num_machines))

# Outer loop to submit jobs
for machine_id in $(seq 0 $((num_machines - 1))); do
    start_index=$((machine_id * molecules_per_machine))
    end_index=$(((machine_id + 1) * molecules_per_machine))

    # Adjust for the last machine
    if [ $machine_id -eq $((num_machines - 1)) ]; then
        end_index=$total_molecules
    fi

    # Calculate the number of batches for this machine
    num_batches=$(((end_index - start_index) / 300 + ((end_index - start_index) % 300 > 0 ? 1 : 0)))

    # Submit job
    bsub -q gpuqueue -n 2 -gpu "num=1:j_exclusive=yes" -R "rusage[mem=8] span[hosts=1]" -W 36:00 \
         -o "logs_$machine_id.stdout" -eo "logs_$machine_id.stderr" \
         python3 screen.py --machine_id $machine_id --num_machines $num_machines
done
