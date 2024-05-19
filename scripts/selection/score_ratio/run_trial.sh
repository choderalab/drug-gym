#!/bin/sh

echo "about to start"

cd /home/retchinm/chodera/drug-gym/scripts/selection/score_ratio

/home/retchinm/miniconda3/envs/chodera/bin/python3 /home/retchinm/chodera/drug-gym/scripts/selection/score_ratio/selection_score_ratio.py --score_ratio $1 --out_dir $2

echo "finished"