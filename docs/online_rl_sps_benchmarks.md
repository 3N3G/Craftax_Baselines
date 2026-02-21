# Online RL + LLM SPS Benchmarks

Collected on Babel cluster while validating `online_rl_llm/online_rl_hidden_jax.py`.

## 8 env benchmarks (num_steps=125 unless noted)

- `skip_n=1`: ~32-42 SPS (job `6418029`, updates 10-20)
- `skip_n=5`: ~171-180 SPS (job `6418030`, updates 40-80)
- `skip_n=25`: ~403-421 SPS (job `6418031`, updates 70-90)
- `skip_n=125`: ~563-601 SPS (job `6418032`, updates 40-90)

## 128 env benchmarks

- `skip_n=25`: ~632 SPS at early steady updates (job `6418549`, update 10)
- `skip_n=100000000` (infinity proxy):
  - ~3300-3500 SPS after warmup (job `6418550`, updates 20-50)
  - ~1300 SPS at first logged update in later rerun (job `6418570`, update 10)

## Notes

- Throughput depends heavily on `skip_n` and warmup phase (vLLM + JAX compile cost).
- `skip_n` values that divide `num_steps` avoid pathological varying-scan recompiles.
- Current sweep launcher defaults: `skip_n in {1, 5, 25, 100000000}`.
