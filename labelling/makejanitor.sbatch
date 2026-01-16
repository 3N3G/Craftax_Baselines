#!/bin/bash

#SBATCH --job-name=craft_janitor     # A unique name for the janitor
#SBATCH --partition=preempt          # Use the same partition
#SBATCH --qos=preempt_qos            # Use the same QOS

#SBATCH --output=janitor_%A.out      # Log file for standard output
#SBATCH --error=janitor_%A.err       # Log file for standard error

#SBATCH --ntasks=1                   # Just one task
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G                    
#SBATCH --time=72:00:00


echo "Starting Janitor on node $(hostname)"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "test"
cd /home/geney/Craftax_Baselines/labelling

srun --cpu-bind=none python janitor.py --name vansh --host login3 --port 1111 
