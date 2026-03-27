# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260319_191206_q35_27b_hf_single_overlay4`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_5_27b_hf_hidden | Qwen/Qwen3.5-27B | predict_history_k5 | 1 | 1 | 0 | 29.6881 | 0.0337 | 77.0868 | 0.0130 | 32.0 |
| qwen3_5_27b_hf_hidden | Qwen/Qwen3.5-27B | predict_state_only | 1 | 1 | 0 | 4.6789 | 0.2137 | 8.1472 | 0.1227 | 32.0 |
