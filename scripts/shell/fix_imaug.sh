#!/bin/bash
# Fix imaug environment: install JAX 0.8.0 (matching craftax)

set -e

echo "=== Fixing imaug environment ==="

# Activate imaug
echo "Activating imaug..."
source ~/.bashrc
conda activate /data/user_data/geney/.conda/envs/imaug

# Remove broken JAX packages
echo "Removing broken JAX packages..."
pip uninstall -y jax jaxlib jax-cuda12-pjrt jax-cuda13-pjrt 2>/dev/null || true

# Install JAX 0.8.0 (matching craftax env)
echo "Installing JAX 0.8.0 and jaxlib 0.8.0..."
pip install jax==0.8.0 jaxlib==0.8.0

# Verify
echo "=== Verifying installation ==="
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
"

echo "=== Testing all required imports ==="
python -c "
import redis
import torch
import transformers
import wandb
import jax
from craftax.craftax.renderer import render_craftax_text
print('All imports successful!')
"

echo "=== imaug environment fixed! ==="
