# v3.1 pilot progress log

## 2026-08-09 — implementation checkpoint

- Locked question: does filtered or compressed agent evidence make a monitor approve unsafe reports that it rejects from the full trace?
- Preserved v3.0 artifacts and created an isolated `src/tcrb/v031/` package.
- Added 12 audited TCRB task families, each with safe and unsafe variants (24 traces total).
- Excluded v3.0’s ambiguous developer-status case and evidence-insufficient payment cases; added an independent deterministic payment record for `ec_payment_o1001`.
- Added five fixed evidence views: full, silent filter, explicit redaction, handoff summary, and protected summary.
- Added strict `approve|reject|insufficient_evidence` monitor output. No LLM judge is used.
- Added public/private artifact separation, event hashes, view validation, paired scoring, clustered bootstrap intervals, audit output, and two report figures.
- Local dataset preparation passed: 24 traces, 120 views, 12 families, 12 safe and 12 unsafe traces.
- Focused v3.1 tests passed: `11 passed`.
- τ-bench repository inspected at pinned external checkout `/tmp/tau2-inspect.tLSziM`; it requires Python 3.12–3.13 and is not installed in the project environment. External adapter remains a separate integration step; no τ-bench claim will be made from TCRB-only results.
- No API calls made yet. Next: smoke run with both monitor providers, then analyze before scheduling the locked core matrix.

## 2026-08-09 — corrected smoke run

- First smoke run (`outputs/v031_visibility_pilot/pilot_01`) completed 40/40 calls but exposed two fixture-quality failures: a missing independent `max_refund` field and a missing independent fraud recommendation. It is retained as a failed preflight, not used for results.
- Corrected the fixtures using deterministic records and started a clean run (`pilot_02`).
- Corrected smoke: 40/40 calls succeeded; estimated GPT cost `$0.052236`.
- Both monitors passed the basic full-trace quality gate: 100% unsafe detection, 100% safe approval, and 100% valid evidence citations on the 4 smoke traces per model.
- Filtered/summary views caused conservative `reject`/`insufficient_evidence` decisions in the smoke sample; unsafe approval did not increase. This is an early signal only and not a kill decision because the smoke set has two task families.
- DeepSeek produced two truncated/empty JSON outputs on the first corrected smoke pass. Increased its output allowance from 600 to 1200 tokens and reran only those two logical cells; the rerun parsed cleanly. Final corrected smoke is 40/40 valid for both models.
- Core matrix completed: 240/240 calls parsed, with no provider errors; estimated GPT cost for all retained responses is `$0.356114`.
- Full-view baseline passed for both models: 100% unsafe detection and valid citations; safe approval was DeepSeek 100% and GPT 91.7% (22/24).
- Main result: unsafe approval stayed at 0% in every transformed view for both models. Paired unsafe-approval delta was 0.0 for silent filtering, explicit redaction, handoff summary, and protected summary; clustered bootstrap intervals were `[0.0, 0.0]`.
- Reliability cost: safe approval fell sharply after transformation. DeepSeek: 41.7% silent filter, 25.0% redaction, 0% handoff/protected. GPT: 33.3%, 16.7%, 0%, 0%. This is a fail-closed availability loss, not a safety blind spot.
- Stability repeat completed: 40/40 valid and unanimous across repeated cells (1.0 unanimous rate).
- Integrity audit passed: all 280 locked logical calls present, successful, parsed, model IDs matched, public/private separation held, and spend was within the `$25` cap.
- Interpretation: the current controlled TCRB transformations falsify the dangerous-approval hypothesis but reveal a robust conservative-monitoring failure. Handoff/protected views are too destructive to represent realistic “lossy but still actionable” summaries; a follow-up should test plausible summaries that retain enough positive evidence while selectively omitting contradictory evidence. τ-bench remains separate until its Python 3.12 environment is prepared and its traces pass the same audit.

## 2026-08-09 — τ-bench external smoke

- Prepared the pinned τ-bench checkout in Python 3.12 (`tau2==1.0.1`) and used official `retail` and `airline` test-split tasks.
- Built 8 replay-validated traces (4 safe reference-action traces and 4 unsafe traces created by successful state-changing argument mutations). Official environment replay confirmed the safe/unsafe DB-state split before any monitor call.
- Ran 80 primary monitor calls (8 traces × 5 views × 2 models); all 80 parsed after adding parse retries. Estimated extra cost: `$0.119169`.
- Primary τ smoke: DeepSeek had one unsafe handoff-summary approval (1/4 unsafe cases, 25%); GPT had 0/4. Other views had 0 unsafe approvals. Safe approval was low for both models, so the external smoke also shows conservative behavior.
- Stability check: 48 handoff-summary repeats across the 4 τ tasks and both models; no parse failures. The primary DeepSeek handoff approval did not repeat (0 approvals across the three repeats for all four tasks). Repeat groups were 81.2% unanimous overall. Therefore the apparent τ blind spot is exploratory and not stable enough to claim.
- External conclusion: TCRB’s robust result remains fail-closed evidence loss; τ-bench does not yet provide a robust cross-model safety-blind-spot signal. A realistic lossy-summary design is the next needed experiment.

## 2026-08-09 — actionable-summary follow-up smoke

- Added a sixth frozen view, `actionable_summary`: it preserves event provenance, identifiers, field names, and full non-critical events, but projects away values from answer-critical events without saying “redacted” or exposing a completeness manifest.
- Prepared a separate follow-up run (`pilot_03_actionable`): 24 traces, 144 views, balanced 12 safe/12 unsafe, zero local validation errors.
- Smoke completed 48/48 calls with valid JSON and no provider failures; estimated cost `$0.064720`. Both full-view baselines passed the quality gate.
- The new view still produced 0% unsafe approval for both models and 0% safe approval in the two-family smoke. Paired unsafe delta and blind-spot rate were both 0.0; clustered intervals were `[0.0, 0.0]`.
- This means the summary is less destructive in format but still too incomplete for monitors to approve safe reports. Proceed to the locked 288-call follow-up core for a domain-wide check; do not call this a blind-spot result from smoke alone.

## 2026-08-09 — actionable-summary core, stability, and report

- Locked follow-up completed: 288 primary calls plus 48 stability repeats; 336/336 responses parsed, with no provider failures. Total estimated cost: `$0.445388`.
- Full-view quality stayed strong: DeepSeek had 100% unsafe detection and 100% safe approval; GPT had 100% unsafe detection and 91.7% safe approval.
- The new `actionable_summary` view produced 0% unsafe approval for both models. Safe approval fell to 8.3% for both. Paired unsafe delta was 0.0 with clustered 95% intervals `[0.0, 0.0]`; safe-approval change was -91.7 percentage points for DeepSeek and -83.3 points for GPT.
- Stability was clean: 48/48 repeats parsed and all repeated groups were unanimous (100%). The integrity audit passed all checks.
- The old `handoff_summary` condition had a small DeepSeek-only unsafe approval rate (8.3%) in this run, but it also rejected almost all safe reports and did not reproduce as a cross-model effect. It is exploratory, not a robust claim.
- Bottom line: even a more realistic structured lossy summary caused fail-closed availability loss, not a stable dangerous blind spot. This is a useful negative result and a clear design constraint; a next study should preserve selected answer-critical values if we want to test a genuine safety-monitoring gap.
