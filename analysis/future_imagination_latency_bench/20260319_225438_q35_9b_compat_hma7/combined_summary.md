# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_225438_q35_9b_compat_hma7`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_vllm_hidden | configs/vllm_hidden_qwen35_9b_auto4 | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_5_9b_vllm_hidden | configs/vllm_hidden_qwen35_9b_auto4 | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
