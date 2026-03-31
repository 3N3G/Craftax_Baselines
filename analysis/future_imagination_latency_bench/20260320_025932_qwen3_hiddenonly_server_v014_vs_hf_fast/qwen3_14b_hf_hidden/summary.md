# Future Imagination Latency Benchmark

## Run Meta

- provider: `hf_local`
- model: `Qwen/Qwen3-14B`
- run_label: `qwen3_14b_hf_hidden`
- num_prompts: `6`
- warmup_prompts_per_variant: `1`
- benchmark_mode: `hidden_only`
- max_tokens: `1`
- effective_max_tokens: `1`
- temperature: `0.4`
- effective_temperature: `0.0`
- top_p: `0.95`
- effective_top_p: `1.0`
- repeats: `3`
- prompt_dir_state: `analysis/future_imagination_latency_bench/prompts_single/predict_state_only`
- prompt_dir_history: `analysis/future_imagination_latency_bench/prompts_single/predict_history_k5`
- hidden_states_path: ``
- hidden_target_layer: `-1`
- hf_model_id: `Qwen/Qwen3-14B`
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
- vllm_enable_prefix_caching: ``
- vllm_enable_chunked_prefill: ``

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_request_calls_per_second | average_hidden_ready_latency_seconds | average_hidden_ready_calls_per_second | average_prompt_tokens | average_completion_tokens | average_total_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-14B | predict_history_k5 | 3 | 3 | 0 | 2.2314 | 0.4482 | 2.2314 | 0.4482 | 6395.0 | 0.0 | 6395.0 |
| Qwen/Qwen3-14B | predict_state_only | 3 | 3 | 0 | 0.4904 | 2.0391 | 0.4904 | 2.0391 | 1626.0 | 0.0 | 1626.0 |
