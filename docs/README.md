# Documentation

## Guides

| Guide | Description |
|-------|-------------|
| [LLM Labelling](llm_labelling.md) | Complete workflow for generating and labelling trajectory data |
| [Symbolic to Pixels](symbolic_to_pixels.md) | How to render symbolic observations for evaluation |

## Key Files

### Data Generation
- `run_ppo_symbolic.sbatch` - Generate symbolic trajectory data

### Labelling (in `labelling/`)
- `llm_worker.py` - LLM labelling worker
- `addtoqueue_llm.py` - Queue manager
- `makeworkers_llm.sbatch` - Worker deployment

### Legacy VLM (in `labelling/`)
- `preempt_safe_worker.py` - VLM labelling worker
- `addtoqueue.py` - Original queue manager
- `makeworkers.sbatch` - VLM worker deployment
