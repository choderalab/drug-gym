# Update to force git pull
#!/bin/sh

echo "about to start"

cd /home/retchinm/chodera/drug-gym/scripts/selection/max_noise

/home/retchinm/miniconda3/envs/chodera/bin/python3 /home/retchinm/chodera/drug-gym/scripts/selection/max_noise/selection_max_noise.py --experiment_state_path $1 --out_dir $2

echo "finished"