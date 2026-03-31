# Qwen3 HF vs Patched vLLM Native (Hidden-State Bench)

- hf_source: `analysis/future_imagination_latency_bench/20260319_234839_qwen3_native_vs_hf`
- vllm_source: `analysis/future_imagination_latency_bench/20260319_235750_qwen3_vllm_only_patched`

| model | prompt_variant | hf_req_s | vllm_req_s | hf_req_per_s | vllm_req_per_s | hf_e2e_s | vllm_e2e_s | faster_req_backend | speedup_x |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Qwen/Qwen3-14B | predict_history_k5 | 13.4996 | 53.4684 | 0.0741 | 0.0187 | 15.3877 | 53.4698 | hf_local | 3.961 |
| Qwen/Qwen3-14B | predict_state_only | 11.6590 | 16.8478 | 0.0858 | 0.0594 | 11.9092 | 16.8483 | hf_local | 1.445 |
| Qwen/Qwen3-4B | predict_history_k5 | 5.6439 | 15.2349 | 0.1772 | 0.0656 | 6.2781 | 15.2365 | hf_local | 2.699 |
| Qwen/Qwen3-4B | predict_state_only | 5.1883 | 7.1106 | 0.1927 | 0.1406 | 5.2722 | 7.1112 | hf_local | 1.371 |
| Qwen/Qwen3-8B | predict_history_k5 | 8.1429 | 27.7353 | 0.1228 | 0.0361 | 8.9909 | 27.7400 | hf_local | 3.406 |
| Qwen/Qwen3-8B | predict_state_only | 7.0361 | 10.5318 | 0.1421 | 0.0950 | 7.1777 | 10.5323 | hf_local | 1.497 |
