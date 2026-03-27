# Future Imagination Latency Benchmark

## Run Meta

- provider: `openai_compatible`
- model: `configs/vllm_hidden_qwen3_14b_auto4`
- run_label: `qwen3_14b_vllm_hidden`
- num_prompts: `14`
- max_tokens: `256`
- temperature: `0.4`
- top_p: `0.95`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts/predict_state_only`
- prompt_dir_history: `analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts/predict_history_k5`
- hidden_states_path: `/scratch/geney/latbench_20260318_172838/hidden_qwen3_14b_vllm_hidden`
- hidden_target_layer: `-1`

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| configs/vllm_hidden_qwen3_14b_auto4 | predict_history_k5 | 7 | 7 | 0 | 13.0448 | 0.0767 | 13.0455 | 0.0767 | 256.0 |
| configs/vllm_hidden_qwen3_14b_auto4 | predict_state_only | 7 | 7 | 0 | 11.8054 | 0.0847 | 11.8058 | 0.0847 | 256.0 |
