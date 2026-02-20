# Documentation

## Guides

| Guide | Description |
|-------|-------------|
| [LLM Labelling](llm_labelling.md) | Complete workflow for generating and labelling trajectory data |
| [Online RL Optimization](online_rl_optimization.md) | Architecture analysis and optimization plan for online RL with LLM hidden states |
| [Progress Journal](progress_journal.md) | Running log of experiments and results |
| [Symbolic to Pixels](symbolic_to_pixels.md) | How to render symbolic observations for evaluation |

## Key Files

### Data Generation
- `online_rl/ppo.py` - Generate symbolic trajectory data (with `--save_trajectory` flag)

### Labelling (in `labelling/`)
- `llm_worker.py` - vLLM-based labelling worker (extracts last-token hidden states)
- `addtoqueue_llm.py` - Queue manager
- `janitor_llm.py` - Re-queue failed jobs
- `obs_to_text.py` - Decode symbolic observations to text
- `add_text_obs.py` - Add text_obs to existing NPZ files
- `makeworkers_llm.sbatch` - Worker deployment (starts vLLM + runs worker)
- `run_labelling.sbatch` - End-to-end coordinator
- `run_extract_hidden.sbatch` - Re-extract hidden states for existing data
