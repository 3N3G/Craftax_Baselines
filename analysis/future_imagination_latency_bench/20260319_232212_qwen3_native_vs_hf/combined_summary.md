# Combined Latency Benchmark

- output_dir: 

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 1 | 0 | 5.7244 | 0.1747 | 6.7622 | 0.1479 | 256.0 |
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 1 | 0 | 5.2527 | 0.1904 | 5.3394 | 0.1873 | 256.0 |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
