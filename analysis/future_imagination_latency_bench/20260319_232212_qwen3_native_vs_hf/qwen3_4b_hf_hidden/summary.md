# Future Imagination Latency Benchmark

## Run Meta

- provider: `hf_local`
- model: `Qwen/Qwen3-4B`
- run_label: `qwen3_4b_hf_hidden`
- num_prompts: `2`
- max_tokens: `256`
- temperature: `0.4`
- top_p: `0.95`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination_latency_bench/prompts_single/predict_state_only`
- prompt_dir_history: `analysis/future_imagination_latency_bench/prompts_single/predict_history_k5`
- hidden_states_path: ``
- hidden_target_layer: `-1`
- hf_model_id: `Qwen/Qwen3-4B`
- hf_device_map: `auto`
- hf_dtype: `bfloat16`
- hf_attn_implementation: ``
- hf_trust_remote_code: `True`
- vllm_model_id: ``
- vllm_tensor_parallel_size: ``
- vllm_dtype: ``
- vllm_max_model_len: ``
- vllm_gpu_memory_utilization: ``
- vllm_trust_remote_code: ``
- vllm_enforce_eager: ``
- vllm_spec_num_tokens: ``

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-4B | predict_history_k5 | 1 | 1 | 0 | 5.7244 | 0.1747 | 6.7622 | 0.1479 | 256.0 |
| Qwen/Qwen3-4B | predict_state_only | 1 | 1 | 0 | 5.2527 | 0.1904 | 5.3394 | 0.1873 | 256.0 |
