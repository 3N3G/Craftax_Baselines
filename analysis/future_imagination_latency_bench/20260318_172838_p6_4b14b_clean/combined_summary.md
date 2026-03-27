# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260318_172838_p6_4b14b_clean`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_14b_vllm_hidden | configs/vllm_hidden_qwen3_14b_auto4 | predict_history_k5 | 7 | 7 | 0 | 13.0448 | 0.0767 | 13.0455 | 0.0767 | 256.0 |
| qwen3_14b_vllm_hidden | configs/vllm_hidden_qwen3_14b_auto4 | predict_state_only | 7 | 7 | 0 | 11.8054 | 0.0847 | 11.8058 | 0.0847 | 256.0 |
| qwen3_4b_vllm_hidden | configs/vllm_hidden_qwen4b | predict_history_k5 | 7 | 7 | 0 | 5.0366 | 0.1985 | 5.0374 | 0.1985 | 256.0 |
| qwen3_4b_vllm_hidden | configs/vllm_hidden_qwen4b | predict_state_only | 7 | 7 | 0 | 3.9230 | 0.2549 | 3.9236 | 0.2549 | 246.3 |
