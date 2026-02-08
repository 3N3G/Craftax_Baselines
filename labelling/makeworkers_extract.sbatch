#!/bin/bash
#SBATCH --job-name=extract_hidden
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --array=1-64
#SBATCH --output=/home/geney/Craftax_Baselines/logs/extract_hidden_%A_%a.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/extract_hidden_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

###############################################################################
# FAST HIDDEN STATE EXTRACTION WORKER
#
# This processes files that already have text_generated (from vLLM) and adds
# hidden_state via fast prefill instead of regeneration.
#
# 10-30x faster than regenerating!
###############################################################################

echo "Starting hidden state extractor on $(hostname)"

source ~/.bashrc
conda activate test  # Has flash_attn 2.8.3

cd /home/geney/Craftax_Baselines/labelling

srun --cpu-bind=none python extract_hidden_states.py --queue-mode \
    --input-dir /data/group_data/rl/geney/craftax_llm_labelled_results \
    --output-dir /data/group_data/rl/geney/new_craftax_llm_labelled_results \
    --batch-size 16
