# Future Imagination Latency Benchmark

## Run Meta

- provider: `hf_local`
- model: `Qwen/Qwen3.5-9B`
- run_label: `qwen3_5_9b_hf_hidden`
- num_prompts: `2`
- max_tokens: `32`
- temperature: `0.4`
- top_p: `0.95`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination_latency_bench/prompts_single/predict_state_only`
- prompt_dir_history: `analysis/future_imagination_latency_bench/prompts_single/predict_history_k5`
- hidden_states_path: ``
- hidden_target_layer: `-1`
- hf_model_id: `Qwen/Qwen3.5-9B`
- hf_device_map: `auto`
- hf_dtype: `bfloat16`
- hf_attn_implementation: ``
- hf_trust_remote_code: `True`

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B | predict_history_k5 | 1 | 1 | 0 | 23.5356 | 0.0425 | 55.7287 | 0.0179 | 32.0 |
| Qwen/Qwen3.5-9B | predict_state_only | 1 | 1 | 0 | 1.6121 | 0.6203 | 2.1198 | 0.4718 | 32.0 |
