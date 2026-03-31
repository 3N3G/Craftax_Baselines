# Combined Latency Benchmark

- output_dir: `analysis/future_imagination_latency_bench/20260318_163847_p6_4b14b_dbg`

| run | model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3_14b_vllm_hidden | configs/vllm_hidden_qwen3_14b_auto4 | predict_history_k5 | 7 | 7 | 0 | 13.0472 | 0.0766 | 13.0484 | 0.0766 | 256.0 |
| qwen3_14b_vllm_hidden | configs/vllm_hidden_qwen3_14b_auto4 | predict_state_only | 7 | 7 | 0 | 11.8045 | 0.0847 | 11.8050 | 0.0847 | 256.0 |
| qwen3_4b_vllm_hidden | configs/vllm_hidden_qwen4b | predict_history_k5 | 7 | 7 | 0 | 4.9079 | 0.2038 | 4.9086 | 0.2037 | 256.0 |
| qwen3_4b_vllm_hidden | configs/vllm_hidden_qwen4b | predict_state_only | 7 | 7 | 0 | 3.9154 | 0.2554 | 3.9158 | 0.2554 | 246.3 |
