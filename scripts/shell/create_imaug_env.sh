#!/bin/bash
# Script to create "imaug" conda environment
# Combines: test env packages + craftax JAX versions

set -e

echo "=== Step 1: Export both environments ==="
conda activate /data/user_data/geney/.conda/envs/test
pip freeze > /tmp/test_packages.txt
conda list > /tmp/test_conda.txt

conda activate /data/user_data/geney/.conda/envs/craftax
pip freeze > /tmp/craftax_packages.txt
pip show jax jaxlib > /tmp/craftax_jax_versions.txt

echo ""
echo "=== JAX versions in craftax (working): ==="
cat /tmp/craftax_jax_versions.txt
echo ""

echo "=== Step 2: Create new imaug environment ==="
# Clone test env as base
conda create -n imaug --clone /data/user_data/geney/.conda/envs/test -y

echo "=== Step 3: Fix JAX in imaug ==="
conda activate imaug

# Remove broken JAX packages
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda13-plugin 2>/dev/null || true

# Get JAX version from craftax
JAX_VERSION=$(conda run -n /data/user_data/geney/.conda/envs/craftax pip show jax | grep Version | awk '{print $2}')
JAXLIB_VERSION=$(conda run -n /data/user_data/geney/.conda/envs/craftax pip show jaxlib | grep Version | awk '{print $2}')

echo "Installing JAX version: $JAX_VERSION"
echo "Installing jaxlib version: $JAXLIB_VERSION"

# Install matching versions
pip install jax==$JAX_VERSION jaxlib==$JAXLIB_VERSION

echo ""
echo "=== Step 4: Verify ==="
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

echo ""
echo "=== Done! ==="
echo "New environment: imaug"
echo "Activate with: conda activate imaug"
