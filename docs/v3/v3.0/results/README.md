# TCRB v0.3 evidence-provenance pilot results

This folder is the complete shareable result package for the fixed two-model pilot.

## What ran

- 16 deterministic simulated base cases across four domains
- 4 matched variants per case: correct/corrupt payload × verified/warning self-label
- 2 prompts: baseline and evidence-first
- 2 models: DeepSeek V4 Flash and GPT-5.6 Terra
- 256 core calls plus 128 repeat calls
- Estimated paid GPT cost: $0.398920

The raw trace data is simulated. The model responses in `responses.jsonl` are recorded API outputs. `scores.jsonl` contains the 384 current locked logical results. The raw file also retains 35 superseded errors/retries for auditability.

## Start here

1. `figures/pilot-brief.md` — one-page result
2. `figures/figure-1-label-evidence-interaction.png` — direct 2×2 behavior
3. `figures/figure-2-prompt-improvement.png` — paired changes with uncertainty
4. `figures/figure-3-repeatability.png` — three-run stability check
5. `trajectory-walkthrough.md` — exact trace and recorded decisions
6. `audit.md` — integrity checks and artifact hashes

## Reproduce analysis

```bash
PYTHONPATH=src python3 -m tcrb.v03 --run-dir docs/v3/results analyze
PYTHONPATH=src python3 -m tcrb.v03 --run-dir docs/v3/results report --output-dir docs/v3/results/figures
PYTHONPATH=src python3 -m tcrb.v03 --run-dir docs/v3/results audit
```

The pre-registered outreach gate did not pass. Treat this as a small, model-specific pilot and a promising test of evidence-first prompting, not proof of a broad failure mode.
