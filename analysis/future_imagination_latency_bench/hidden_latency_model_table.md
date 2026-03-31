# Hidden-State Latency Model Table

- benchmark_root: `analysis/future_imagination_latency_bench`
- model_matrix: `configs/future_imagination/hidden_latency_model_matrix.json`
- `request_avg_s`: average latency for the model/backend call itself.
- `request_avg_responses_per_second`: inverse of `request_avg_s`.
- `hidden_ready_avg_s`: end-to-end latency until the hidden vector is readable by the benchmark.
- `hidden_ready_avg_responses_per_second`: inverse of `hidden_ready_avg_s`.

| model_name | category | backend | benchmark_mode | status | state_prompt_tokens | state_output_tokens | state_request_avg_s | state_request_avg_responses_per_second | state_hidden_ready_avg_s | state_hidden_ready_avg_responses_per_second | history_prompt_tokens | history_output_tokens | history_request_avg_s | history_request_avg_responses_per_second | history_hidden_ready_avg_s | history_hidden_ready_avg_responses_per_second | total_calls | successful_calls | error_calls | source_run_dir |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-4B-Thinking-2507 (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | ok | 1626.0 | 1.0 | 0.0278 | 35.9385 | 0.0280 | 35.6697 | 6395.0 | 1.0 | 0.0444 | 22.5085 | 0.0459 | 21.7787 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_4b_vllm_hidden |
| Qwen3-4B (HF) | transformers | hf_local | hidden_only | ok | 1626.0 | 0.0 | 0.1274 | 7.8465 | 0.1274 | 7.8465 | 6395.0 | 0.0 | 0.6795 | 1.4716 | 0.6795 | 1.4716 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_4b_hf_hidden |
| Qwen3-8B (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | ok | 1626.0 | 1.0 | 0.0389 | 25.7347 | 0.0391 | 25.5864 | 6395.0 | 1.0 | 0.0570 | 17.5334 | 0.0587 | 17.0321 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_8b_vllm_hidden |
| Qwen3-8B (HF) | transformers | hf_local | hidden_only | ok | 1626.0 | 0.0 | 0.2518 | 3.9710 | 0.2518 | 3.9710 | 6395.0 | 0.0 | 1.2316 | 0.8120 | 1.2316 | 0.8120 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_8b_hf_hidden |
| Qwen3-14B (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | ok | 1626.0 | 1.0 | 0.0606 | 16.4990 | 0.0609 | 16.4152 | 6395.0 | 1.0 | 0.0783 | 12.7722 | 0.0799 | 12.5175 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_14b_vllm_hidden |
| Qwen3-14B (HF) | transformers | hf_local | hidden_only | ok | 1626.0 | 0.0 | 0.4904 | 2.0391 | 0.4904 | 2.0391 | 6395.0 | 0.0 | 2.2314 | 0.4482 | 2.2314 | 0.4482 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_14b_hf_hidden |
| Qwen3.5-9B (vLLM Native Offline) | vllm_compatible | vllm_offline | hidden_only | ok | 1662.0 | 1.0 | 0.1408 | 7.1034 | 0.1412 | 7.0813 | 6436.0 | 1.0 | 0.5235 | 1.9101 | 0.5241 | 1.9082 | 2 | 2 | 0 | future_imagination_latency_bench/20260321_002450_q35_9b_vllm_offline_splitfix5_rl/qwen3_5_9b_vllm_offline_hidden |
| Qwen3.5-9B (HF) | transformers | hf_local | hidden_only | ok | 1662.0 | 0.0 | 0.3414 | 2.9287 | 0.3414 | 2.9287 | 6436.0 | 0.0 | 1.4591 | 0.6854 | 1.4591 | 0.6854 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_032626_qwen35_9b_plugin_hf_hiddenonly_fast3/qwen3_5_9b_hf_hidden |
| Qwen3.5-27B (vLLM Native Offline) | vllm_compatible | vllm_offline |  | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Qwen3.5-27B (HF) | transformers | hf_local | hidden_only | ok | 1662.0 | 0.0 | 1.8956 | 0.5275 | 1.8956 | 0.5275 | 6436.0 | 0.0 | 7.3815 | 0.1355 | 7.3815 | 0.1355 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_032825_qwen35_27b_plugin_hf_hiddenonly_fast3/qwen3_5_27b_hf_hidden |

