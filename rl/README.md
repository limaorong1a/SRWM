# RL Data Collection (SFT Warmup)

This folder contains scripts to collect successful ALFWorld trajectories with a
closed-source model served behind your company Dify-compatible API, then
convert those trajectories into SFT JSONL data.

## Scope of this stage

- Build and test SFT data collection pipeline only.
- Do not start GRPO training in this stage.
- Keep all secrets outside git-tracked files.

## Environment variables

Set these on the Linux server before running:

- `DIFY_ACCESS_KEY_ID`
- `DIFY_ACCESS_KEY_SECRET`
- `DIFY_ALFWORLD_AGENT_ID`
- `ALFWORLD_DATA`

## Quick start

1. Prepare environment:

```bash
bash rl/scripts/setup_env.sh
```

2. Smoke test (single trajectory):

```bash
bash rl/scripts/run_collect_sft.sh smoke
```

3. Pilot run (10 successes per task type target):

```bash
bash rl/scripts/run_collect_sft.sh pilot
```

4. Full run (200 successes per task type target):

```bash
bash rl/scripts/run_collect_sft.sh full
```

5. Convert won trajectories to SFT JSONL:

```bash
python -m rl.data.format_to_sft_jsonl --input rl/data/raw --output rl/data/sft/dataset.jsonl
```

6. Generate collection report:

```bash
python -m rl.data.stats_report --raw rl/data/raw --output rl/data/sft/stats_report.json
```

## Notes

- Raw trajectories include both success and failure cases.
- SFT dataset includes only `won=True` trajectories.
- Resume mode is supported via progress files in `rl/data/raw/`.

