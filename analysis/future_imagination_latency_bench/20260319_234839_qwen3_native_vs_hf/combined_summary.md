# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_234839_qwen3_native_vs_hf`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_14b_hf_hidden | Qwen/Qwen3-14B | predict_history_k5 | 1 | 1 | 0 | 13.4996 | 0.0741 | 15.3877 | 0.0650 | 256.0 |
| qwen3_14b_hf_hidden | Qwen/Qwen3-14B | predict_state_only | 1 | 1 | 0 | 11.6590 | 0.0858 | 11.9092 | 0.0840 | 256.0 |
| qwen3_14b_vllm_native_hidden | Qwen/Qwen3-14B | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_14b_vllm_native_hidden | Qwen/Qwen3-14B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 1 | 0 | 5.6439 | 0.1772 | 6.2781 | 0.1593 | 256.0 |
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 1 | 0 | 5.1883 | 0.1927 | 5.2722 | 0.1897 | 256.0 |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_8b_hf_hidden | Qwen/Qwen3-8B | predict_history_k5 | 1 | 1 | 0 | 8.1429 | 0.1228 | 8.9909 | 0.1112 | 256.0 |
| qwen3_8b_hf_hidden | Qwen/Qwen3-8B | predict_state_only | 1 | 1 | 0 | 7.0361 | 0.1421 | 7.1777 | 0.1393 | 256.0 |
| qwen3_8b_vllm_native_hidden | Qwen/Qwen3-8B | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_8b_vllm_native_hidden | Qwen/Qwen3-8B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
