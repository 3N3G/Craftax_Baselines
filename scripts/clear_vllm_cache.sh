#!/bin/bash
# Clear vLLM torch.compile cache when corrupted

echo "Clearing vLLM torch.compile cache..."
rm -rf ~/.cache/vllm/torch_compile_cache/

echo "Cache cleared. You can now retry your job."
echo ""
echo "To run your job again:"
echo "  sbatch scripts/sbatch/run_online_rl_hidden.sbatch 128 100000000 25"