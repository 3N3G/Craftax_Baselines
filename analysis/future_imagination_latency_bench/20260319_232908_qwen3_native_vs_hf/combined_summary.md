# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_232908_qwen3_native_vs_hf`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 1 | 0 | 8.3134 | 0.1203 | 9.2425 | 0.1082 | 256.0 |
| qwen3_4b_hf_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 1 | 0 | 7.6495 | 0.1307 | 7.7566 | 0.1289 | 256.0 |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 0 | 1 |  |  |  |  |  |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 0 | 1 |  |  |  |  |  |
