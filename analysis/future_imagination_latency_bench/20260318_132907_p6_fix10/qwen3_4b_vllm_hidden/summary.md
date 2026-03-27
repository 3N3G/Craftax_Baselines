# Future Imagination Latency Benchmark

## Run Meta

- provider: `openai_compatible`
- model: `configs/vllm_hidden_qwen4b`
- run_label: `qwen3_4b_vllm_hidden`
- num_prompts: `14`
- max_tokens: `256`
- temperature: `0.4`
- top_p: `0.95`
- repeats: `1`
- prompt_dir_state: `analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts/predict_state_only`
- prompt_dir_history: `analysis/future_imagination/20260316_111259_traj_20260311_100410_predict_stateplushistoryk5_key7_promptrev_qwen3-4b-p6/prompts/predict_history_k5`
- hidden_states_path: `/scratch/geney/latbench_20260318_132907/hidden_qwen3_4b_vllm_hidden`
- hidden_target_layer: `-1`

## Summary

| model | variant | calls | ok | err | req_mean_s | req_p50_s | req_p90_s | hidden_mean_s | total_with_hidden_mean_s | completion_tokens_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| configs/vllm_hidden_qwen4b | predict_history_k5 | 7 | 7 | 0 | 4.0527 | 4.6822 | 4.7257 | 0.0008 | 4.0535 | 232.3 |
| configs/vllm_hidden_qwen4b | predict_state_only | 7 | 7 | 0 | 3.4213 | 3.4944 | 3.5068 | 0.0004 | 3.4217 | 252.6 |
