# Hidden-State Latency Model Table

- benchmark_root: `analysis/future_imagination_latency_bench`
- model_matrix: `configs/future_imagination/hidden_latency_model_matrix.json`
- `request_latency_s`: latency for the model/backend call itself.
- `hidden_ready_latency_s`: end-to-end latency until the hidden vector is readable by the benchmark.
- `hidden_ready_responses_per_second`: inverse of the end-to-end hidden-ready latency.

| model_name | category | backend | benchmark_mode | status | state_prompt_tokens | state_output_tokens | state_request_latency_s | state_hidden_ready_latency_s | state_hidden_ready_responses_per_second | history_prompt_tokens | history_output_tokens | history_request_latency_s | history_hidden_ready_latency_s | history_hidden_ready_responses_per_second | total_calls | successful_calls | error_calls | source_run_dir |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-4B-Thinking-2507 (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | ok | 1626.0 | 1.0 | 0.0278 | 0.0280 | 35.6697 | 6395.0 | 1.0 | 0.0444 | 0.0459 | 21.7787 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_4b_vllm_hidden |
| Qwen3-4B (HF) | transformers | hf_local | hidden_only | ok | 1626.0 | 0.0 | 0.1274 | 0.1274 | 7.8465 | 6395.0 | 0.0 | 0.6795 | 0.6795 | 1.4716 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_4b_hf_hidden |
| Qwen3-8B (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | ok | 1626.0 | 1.0 | 0.0389 | 0.0391 | 25.5864 | 6395.0 | 1.0 | 0.0570 | 0.0587 | 17.0321 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_8b_vllm_hidden |
| Qwen3-8B (HF) | transformers | hf_local | hidden_only | ok | 1626.0 | 0.0 | 0.2518 | 0.2518 | 3.9710 | 6395.0 | 0.0 | 1.2316 | 1.2316 | 0.8120 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_8b_hf_hidden |
| Qwen3-14B (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | ok | 1626.0 | 1.0 | 0.0606 | 0.0609 | 16.4152 | 6395.0 | 1.0 | 0.0783 | 0.0799 | 12.5175 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_14b_vllm_hidden |
| Qwen3-14B (HF) | transformers | hf_local | hidden_only | ok | 1626.0 | 0.0 | 0.4904 | 0.4904 | 2.0391 | 6395.0 | 0.0 | 2.2314 | 2.2314 | 0.4482 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_025932_qwen3_hiddenonly_server_v014_vs_hf_fast/qwen3_14b_hf_hidden |
| Qwen3.5-9B (vLLM Plugin API) | vllm_compatible | openai_compatible | hidden_only | error |  |  |  |  |  |  |  |  |  |  | 6 | 0 | 6 | future_imagination_latency_bench/20260320_032626_qwen35_9b_plugin_hf_hiddenonly_fast3/qwen3_5_9b_vllm_hidden |
| Qwen3.5-9B (HF) | transformers | hf_local | hidden_only | ok | 1662.0 | 0.0 | 0.3414 | 0.3414 | 2.9287 | 6436.0 | 0.0 | 1.4591 | 1.4591 | 0.6854 | 6 | 6 | 0 | future_imagination_latency_bench/20260320_032626_qwen35_9b_plugin_hf_hiddenonly_fast3/qwen3_5_9b_hf_hidden |
| Qwen3.5-27B (vLLM Plugin API) | vllm_compatible | openai_compatible |  | missing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Qwen3.5-27B (HF) | transformers | hf_local |  | ok |  | 32.0 | 4.6789 | 8.1472 | 0.1227 |  | 32.0 | 29.6881 | 77.0868 | 0.0130 | 2 | 2 | 0 | future_imagination_latency_bench/20260319_191206_q35_27b_hf_single_overlay4/qwen3_5_27b_hf_hidden |

