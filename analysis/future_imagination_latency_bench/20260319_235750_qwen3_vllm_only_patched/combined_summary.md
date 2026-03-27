# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_235750_qwen3_vllm_only_patched`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_14b_vllm_native_hidden | Qwen/Qwen3-14B | predict_history_k5 | 1 | 1 | 0 | 53.4684 | 0.0187 | 53.4698 | 0.0187 | 256.0 |
| qwen3_14b_vllm_native_hidden | Qwen/Qwen3-14B | predict_state_only | 1 | 1 | 0 | 16.8478 | 0.0594 | 16.8483 | 0.0594 | 256.0 |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_history_k5 | 1 | 1 | 0 | 15.2349 | 0.0656 | 15.2365 | 0.0656 | 256.0 |
| qwen3_4b_vllm_native_hidden | Qwen/Qwen3-4B | predict_state_only | 1 | 1 | 0 | 7.1106 | 0.1406 | 7.1112 | 0.1406 | 256.0 |
| qwen3_8b_vllm_native_hidden | Qwen/Qwen3-8B | predict_history_k5 | 1 | 1 | 0 | 27.7353 | 0.0361 | 27.7400 | 0.0360 | 256.0 |
| qwen3_8b_vllm_native_hidden | Qwen/Qwen3-8B | predict_state_only | 1 | 1 | 0 | 10.5318 | 0.0950 | 10.5323 | 0.0949 | 256.0 |
