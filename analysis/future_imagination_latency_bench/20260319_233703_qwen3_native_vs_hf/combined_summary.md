# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_233703_qwen3_native_vs_hf`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 1 | 0 | 8.5029 | 0.1176 | 9.4415 | 0.1059 | 256.0 |
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 1 | 0 | 7.8398 | 0.1276 | 7.9339 | 0.1260 | 256.0 |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
