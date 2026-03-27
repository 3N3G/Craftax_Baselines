# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260320_032825_qwen35_27b_plugin_hf_hiddenonly_fast3`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_27b_hf_hidden | Qwen/Qwen3.5-27B | predict_history_k5 | 3 | 3 | 0 | 7.3815 | 0.1355 | 7.3815 | 0.1355 | 0.0 |
| qwen3_5_27b_hf_hidden | Qwen/Qwen3.5-27B | predict_state_only | 3 | 3 | 0 | 1.8956 | 0.5275 | 1.8956 | 0.5275 | 0.0 |
| qwen3_5_27b_vllm_hidden | configs/vllm_hidden_qwen35_27b_auto4 | predict_history_k5 | 3 | 0 | 3 |  |  |  |  |  |
| qwen3_5_27b_vllm_hidden | configs/vllm_hidden_qwen35_27b_auto4 | predict_state_only | 3 | 0 | 3 |  |  |  |  |  |
