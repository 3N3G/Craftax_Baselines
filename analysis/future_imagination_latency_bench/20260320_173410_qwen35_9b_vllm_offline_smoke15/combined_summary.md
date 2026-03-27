# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260320_173410_qwen35_9b_vllm_offline_smoke15`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_vllm_offline_hidden | /data/group_data/rl/geney/model_mirrors/Qwen3.5-9B | predict_history_k5 | 1 | 1 | 0 | 0.5662 | 1.7660 | 0.5667 | 1.7646 | 1.0 |
| qwen3_5_9b_vllm_offline_hidden | /data/group_data/rl/geney/model_mirrors/Qwen3.5-9B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
