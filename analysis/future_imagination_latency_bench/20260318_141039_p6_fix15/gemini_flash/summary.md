# Future Imagination Latency Benchmark

## Run Meta

- provider: `gemini`
- model: `gemini-3-flash-preview`
- run_label: `gemini_flash`
- num_prompts: `14`
- max_tokens: `256`
- temperature: `0.4`
- top_p: `0.95`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts/predict_state_only`
- prompt_dir_history: `analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts/predict_history_k5`
- hidden_states_path: ``
- hidden_target_layer: `-1`

## Summary

| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_responses_per_second | average_end_to_end_latency_seconds | average_end_to_end_responses_per_second | average_completion_tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini-3-flash-preview | predict_history_k5 | 7 | 7 | 0 | 2.0756 | 0.4818 | 2.0756 | 0.4818 | 9.7 |
| gemini-3-flash-preview | predict_state_only | 7 | 7 | 0 | 1.9954 | 0.5012 | 1.9954 | 0.5012 | 10.9 |
