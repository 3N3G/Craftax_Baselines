# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260320_032626_qwen35_9b_plugin_hf_hiddenonly_fast3`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_9b_hf_hidden | Qwen/Qwen3.5-9B | predict_history_k5 | 3 | 3 | 0 | 1.4591 | 0.6854 | 1.4591 | 0.6854 | 0.0 |
| qwen3_5_9b_hf_hidden | Qwen/Qwen3.5-9B | predict_state_only | 3 | 3 | 0 | 0.3414 | 2.9287 | 0.3414 | 2.9287 | 0.0 |
| qwen3_5_9b_vllm_hidden | configs/vllm_hidden_qwen35_9b_auto4 | predict_history_k5 | 3 | 0 | 3 |  |  |  |  |  |
| qwen3_5_9b_vllm_hidden | configs/vllm_hidden_qwen35_9b_auto4 | predict_state_only | 3 | 0 | 3 |  |  |  |  |  |
