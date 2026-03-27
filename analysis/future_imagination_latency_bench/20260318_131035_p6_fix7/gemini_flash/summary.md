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

| model | variant | calls | ok | err | req_mean_s | req_p50_s | req_p90_s | hidden_mean_s | total_with_hidden_mean_s | completion_tokens_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini-3-flash-preview | predict_history_k5 | 7 | 7 | 0 | 2.0238 | 1.9788 | 2.1310 |  | 2.0238 | 9.9 |
| gemini-3-flash-preview | predict_state_only | 7 | 7 | 0 | 2.1038 | 2.1368 | 2.1783 |  | 2.1038 | 10.4 |
