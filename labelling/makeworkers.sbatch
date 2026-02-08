#!/bin/bash

#SBATCH --job-name=craft_labelling     # A single name for all jobs in the array
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos             
#SBATCH --array=1-30                   # Submit 30 tasks, indexed 1 through 30
#SBATCH --output=worker_%A_%a.out      # Log file for standard output. %A = Job ID, %a = Task ID
#SBATCH --error=worker_%A_%a.err       # Log file for standard error.
#
# --- Resource Request ---
#SBATCH --ntasks=1                     # Each task in the array runs 1 task (your python script)
#SBATCH --cpus-per-task=8              # Request 8 CPUs for that one task
#SBATCH --gres=gpu:L40S:1              # Request 1 L40S GPU for that one task
#
# --- Optional ---
#SBATCH --time=24:00:00
#SBATCH --mem=256G


echo "Starting worker on node $(hostname)"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "test"
cd /home/geney/Craftax_Baselines/labelling

srun --cpu-bind=none python run_worker.py --name vansh --host login3 --port 1111 
