# Future Imagination Eval

This workflow evaluates "future imagination" quality on recorded trajectories using prompt templates and external LLM providers.

## Files
- `scripts/future_imagination_eval.py`
- `scripts/render_future_prompt_browser.py`
- `configs/future_imagination/run_config_template.json`
- `configs/future_imagination/templates/*.txt`

## What The Script Produces
Under `analysis/future_imagination/<run_dir>/`:
- `records.jsonl`: full per-call logs (prompt hash, response, usage, errors)
- `selected_states.jsonl`: state snapshots for the selected timesteps
- `records.csv`: scan-friendly table
- `summary_by_run.csv`: aggregate stats per run id
- `pairwise_scores.csv`: lexical similarity vs oracle run id
- `report.md`: readable markdown report
- `report.html`: browser-friendly per-timestep inspection report
- `estimate.json`: rough preflight token/call estimate
- `resolved_run_config.json`: run provenance
- `prompts/<run_id>/t_XXXXX.txt`: full prompt text (when `--store-full-prompts` is enabled)

## Prompt Variables
Available placeholders in prompt templates:
- `{trajectory_id}`
- `{timestep}`
- `{total_timesteps}`
- `{future_snapshot_stride}`
- `{episode_id}`
- `{action_at_t}`
- `{reward_at_t}`
- `{done_at_t}`
- `{current_state_raw}`
- `{current_state_filtered}`
- `{current_state_compact}`
- `{history_block}`
- `{future_state_block}`
- `{future_event_block}`

## How Context Size Works

There are two independent "stride/max" concepts:

1) `selection.stride` + `selection.max_states`
- Controls which timesteps are evaluated at all.
- Example: `selection.stride=25` means evaluate t=0,25,50,... (plus terminal if enabled).
- `selection.max_states` truncates that selected set.
- `selection.mode`:
  - `range_plus_explicit` (default): use range (`start_t..end_t` with `stride`) plus `explicit_timesteps`
  - `explicit_only`: ignore range; use only `explicit_timesteps` (plus terminal if enabled)

2) Per-run `future_stride` + `future_max_states` (oracle-style runs)
- Controls how many future snapshots are inserted into a single prompt.
- `future_stride=25` samples t, t+25, t+50, ...
- `future_max_states=24` caps those future snapshots.
- `future_event_max` separately caps transition-event lines.

### Token Estimate Method
- The estimator uses rough conversion: `estimated_tokens ~= prompt_chars / 4`.
- This is intentionally conservative and model-agnostic.
- True provider tokenization can differ.

### Gemini Output Length Note
- For Gemini runs, `generation.thinking_budget` can materially affect visible output length.
- If `thinking_budget` is high and `max_output_tokens` is modest, hidden thinking tokens can consume most of the budget, producing short visible responses.
- For narrative-style outputs, set:
  - `generation.thinking_budget: 0`
  - and a sufficiently large `generation.max_output_tokens` (for example 700-1500 depending on desired summary length).

## Safety Guard
By default, templates containing `[TODO]` / `<REPLACE...` are rejected.
Use `--allow-placeholder-templates` only for dry-run/testing.

## Example: Gemini Estimate-Only
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
python3 scripts/future_imagination_eval.py \
  --trajectory-dir play_data/trajectory_records/traj_20260311_100410 \
  --config configs/future_imagination/run_config_template.json \
  --provider gemini \
  --model gemini-2.0-flash \
  --estimate-only'
```

## Example: Gemini Run
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
GEMINI_API_KEY="<set-in-shell>" \
python3 scripts/future_imagination_eval.py \
  --trajectory-dir play_data/trajectory_records/traj_20260311_100410 \
  --config configs/future_imagination/run_config_template.json \
  --provider gemini \
  --model gemini-2.0-flash \
  --store-full-prompts \
  --selection-stride 50 \
  --selection-max-states 40'
```

## Prompt Browser (Full Prompt Scrolling)
After a run with `--store-full-prompts`, render one scrollable markdown:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
python3 scripts/render_future_prompt_browser.py \
  --run-dir analysis/future_imagination/<run_dir> \
  --output analysis/future_imagination/<run_dir>/prompt_browser.md'
```

## Static Hosting Bundle (GitHub Pages / Netlify)
To share the combined HTML with working slider images, build a static bundle that rewrites frame paths and copies required PNGs:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
python3 scripts/prepare_future_report_pages_bundle.py \
  --report-dir analysis/future_imagination/<combined_report_dir> \
  --frame-dir play_data/trajectory_records/traj_20260311_100410/render_frames_bs16 \
  --output-dir site/<publish_folder> \
  --overwrite'
```

If using GitHub Pages from `/docs`, copy the bundle:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
mkdir -p docs/future_reports && \
rm -rf docs/future_reports/<publish_folder> && \
cp -R site/<publish_folder> docs/future_reports/<publish_folder>'
```

## One-Script Prompt Iteration (Edit -> Qwen Rerun -> Combined Report)
Use this helper when iterating prompt text repeatedly:
- Script: `scripts/shell/run_future_prompt_iteration_wave.sh`
- It can:
  - optionally rewrite the single-line `Craftax overview` in prediction templates,
  - push templates to Babel,
  - submit Qwen3-4B and Qwen3.5-9B reruns,
  - wait + pull outputs,
  - rebuild a combined unabridged HTML/MD report.

Example:
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
scripts/shell/run_future_prompt_iteration_wave.sh \
  --tag p4 \
  --overview-text "Craftax is a game about exploring dungeons, mining, crafting and fighting enemies."'
```

Useful flags:
- `--skip-qwen35` (run only 4B)
- `--no-wait` (submit jobs and exit)
- `--no-report` (skip combined report rebuild)
- `--run-config <path>` (switch concise/non-concise config)

## Example: Re-run With Qwen via OpenAI-Compatible Endpoint
```bash
zsh -lic 'cd /Users/gene/Documents/Craftax_Baselines && \
python3 scripts/future_imagination_eval.py \
  --trajectory-dir play_data/trajectory_records/traj_20260311_100410 \
  --config configs/future_imagination/run_config_template.json \
  --provider openai_compatible \
  --base-url http://127.0.0.1:8000 \
  --model ./configs/vllm_hidden_qwen4b'
```
