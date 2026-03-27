# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260321_002450_q35_9b_vllm_offline_splitfix5_rl`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_vllm_offline_hidden | /scratch/geney/latbench_6685099/qwen35_9b_compat | predict_history_k5 | 1 | 1 | 0 | 0.5235 | 1.9101 | 0.5241 | 1.9082 | 1.0 |
| qwen3_5_9b_vllm_offline_hidden | /scratch/geney/latbench_6685099/qwen35_9b_compat | predict_state_only | 1 | 1 | 0 | 0.1408 | 7.1034 | 0.1412 | 7.0813 | 1.0 |
