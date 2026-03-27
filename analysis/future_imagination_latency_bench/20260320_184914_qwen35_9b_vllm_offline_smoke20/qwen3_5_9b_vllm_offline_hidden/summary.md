# Future Imagination Latency Benchmark

## Run Meta

- provider: `vllm_offline`
- model: `/scratch/geney/latbench_6682601/qwen35_9b_compat`
- run_label: `qwen3_5_9b_vllm_offline_hidden`
- num_prompts: `2`
- warmup_prompts_per_variant: `1`
- benchmark_mode: `hidden_only`
- max_tokens: `256`
- effective_max_tokens: `1`
- temperature: `0.4`
- effective_temperature: `0.0`
- top_p: `0.95`
- effective_top_p: `1.0`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination_latency_bench/prompts_single/predict_state_only`
- prompt_dir_history: `analysis/future_imagination_latency_bench/prompts_single/predict_history_k5`
- hidden_states_path: ``
- hidden_target_layer: `-1`
- hf_model_id: ``
- hf_device_map: ``
- hf_dtype: ``
- hf_attn_implementation: ``
- hf_trust_remote_code: ``
- vllm_model_id: `/scratch/geney/latbench_6682601/qwen35_9b_compat`
- vllm_tensor_parallel_size: `1`
- vllm_dtype: `bfloat16`
- vllm_max_model_len: `8192`
- vllm_gpu_memory_utilization: `0.86`
- vllm_trust_remote_code: `True`
- vllm_enforce_eager: `False`
- vllm_spec_num_tokens: `1`
- vllm_enable_prefix_caching: `False`
- vllm_enable_chunked_prefill: `False`
- vllm_enable_hybrid_kv_cache_manager: `False`
- vllm_kv_connector_module_path: `utils.vllm_hidden_connector`
- vllm_kv_connector_mode: `last_token`

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_request_calls_per_second | average_hidden_ready_latency_seconds | average_hidden_ready_calls_per_second | average_prompt_tokens | average_completion_tokens | average_total_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| /scratch/geney/latbench_6682601/qwen35_9b_compat | predict_history_k5 | 1 | 1 | 0 | 0.5381 | 1.8582 | 0.5387 | 1.8564 | 6436.0 | 1.0 | 6437.0 |
| /scratch/geney/latbench_6682601/qwen35_9b_compat | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |  |  |
