# Future Imagination Latency Benchmark

## Run Meta

- provider: `vllm_offline`
- model: `Qwen/Qwen3-8B`
- run_label: `qwen3_8b_vllm_native_hidden`
- num_prompts: `2`
- max_tokens: `256`
- temperature: `0.4`
- top_p: `0.95`
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
- vllm_model_id: `Qwen/Qwen3-8B`
- vllm_tensor_parallel_size: `1`
- vllm_dtype: `bfloat16`
- vllm_max_model_len: `8192`
- vllm_gpu_memory_utilization: `0.8`
- vllm_trust_remote_code: `True`
- vllm_enforce_eager: `True`
- vllm_spec_num_tokens: `1`
- vllm_enable_prefix_caching: `False`
- vllm_enable_chunked_prefill: `False`

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-8B | predict_history_k5 | 1 | 1 | 0 | 27.7353 | 0.0361 | 27.7400 | 0.0360 | 256.0 |
| Qwen/Qwen3-8B | predict_state_only | 1 | 1 | 0 | 10.5318 | 0.0950 | 10.5323 | 0.0949 | 256.0 |
