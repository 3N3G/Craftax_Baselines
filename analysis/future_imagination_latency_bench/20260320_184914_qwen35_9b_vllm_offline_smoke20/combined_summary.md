# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260320_184914_qwen35_9b_vllm_offline_smoke20`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_vllm_offline_hidden | /scratch/geney/latbench_6682601/qwen35_9b_compat | predict_history_k5 | 1 | 1 | 0 | 0.5381 | 1.8582 | 0.5387 | 1.8564 | 1.0 |
| qwen3_5_9b_vllm_offline_hidden | /scratch/geney/latbench_6682601/qwen35_9b_compat | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
