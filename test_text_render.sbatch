#!/bin/bash
#SBATCH --job-name=test_text_render
#SBATCH --output=/home/geney/Craftax_Baselines/logs/test_text_%j.out
#SBATCH --error=/home/geney/Craftax_Baselines/logs/test_text_%j.err
#SBATCH --partition=rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

echo "=========================================="
echo "Testing Text Rendering in PPO"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/craftax

cd /home/geney/Craftax_Baselines

# Run short test with text saving
TEST_DIR="/data/group_data/rl/geney/test_text_render_$(date +%s)/"
echo "Test output directory: $TEST_DIR"

python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --save_traj \
    --total_timesteps 100000 \
    --num_envs 16 \
    --num_steps 32 \
    --save_traj_every 1 \
    --traj_save_path "$TEST_DIR" \
    --no-use_wandb

echo ""
echo "=== Checking saved files ==="
ls -la "$TEST_DIR"

echo ""
echo "=== Verifying text_obs in saved file ==="
python3 -c "
import numpy as np
import glob

files = sorted(glob.glob('$TEST_DIR/*.npz'))
if not files:
    print('ERROR: No files saved!')
else:
    print(f'Found {len(files)} files')
    data = np.load(files[0], allow_pickle=True)
    print(f'Keys: {data.files}')
    
    if 'text_obs' in data.files:
        text_obs = data['text_obs']
        print(f'text_obs shape: {text_obs.shape}')
        print(f'Non-empty texts: {sum(1 for t in text_obs if t)}/{len(text_obs)}')
        print(f'\\nSample text (first entry):\\n{text_obs[0][:500] if text_obs[0] else \"EMPTY\"}...')
    else:
        print('ERROR: text_obs not found in saved file!')
"

echo ""
echo "=========================================="
echo "Test Complete: $(date)"
echo "=========================================="
