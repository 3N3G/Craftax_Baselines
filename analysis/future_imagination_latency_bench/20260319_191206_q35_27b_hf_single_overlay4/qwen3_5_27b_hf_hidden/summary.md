# Future Imagination Latency Benchmark

## Run Meta

- provider: `hf_local`
- model: `Qwen/Qwen3.5-27B`
- run_label: `qwen3_5_27b_hf_hidden`
- num_prompts: `2`
- max_tokens: `32`
- temperature: `0.4`
- top_p: `0.95`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination_latency_bench/prompts_single/predict_state_only`
- prompt_dir_history: `analysis/future_imagination_latency_bench/prompts_single/predict_history_k5`
- hidden_states_path: ``
- hidden_target_layer: `-1`
- hf_model_id: `Qwen/Qwen3.5-27B`
- hf_device_map: `auto`
- hf_dtype: `bfloat16`
- hf_attn_implementation: ``
- hf_trust_remote_code: `True`

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.5-27B | predict_history_k5 | 1 | 1 | 0 | 29.6881 | 0.0337 | 77.0868 | 0.0130 | 32.0 |
| Qwen/Qwen3.5-27B | predict_state_only | 1 | 1 | 0 | 4.6789 | 0.2137 | 8.1472 | 0.1227 | 32.0 |
