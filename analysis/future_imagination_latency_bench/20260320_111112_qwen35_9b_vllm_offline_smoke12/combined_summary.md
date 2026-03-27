# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260320_111112_qwen35_9b_vllm_offline_smoke12`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_vllm_offline_hidden | Qwen/Qwen3.5-9B | predict_history_k5 | 1 | 1 | 0 | 0.9820 | 1.0183 | 0.9824 | 1.0179 | 1.0 |
| qwen3_5_9b_vllm_offline_hidden | Qwen/Qwen3.5-9B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
