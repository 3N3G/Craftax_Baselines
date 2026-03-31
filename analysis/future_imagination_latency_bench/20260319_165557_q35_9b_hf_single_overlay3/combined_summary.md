# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_165557_q35_9b_hf_single_overlay3`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_hf_hidden | Qwen/Qwen3.5-9B | predict_history_k5 | 1 | 1 | 0 | 23.5356 | 0.0425 | 55.7287 | 0.0179 | 32.0 |
| qwen3_5_9b_hf_hidden | Qwen/Qwen3.5-9B | predict_state_only | 1 | 1 | 0 | 1.6121 | 0.6203 | 2.1198 | 0.4718 | 32.0 |
