# Reliability Study: Forward Plan

This plan continues one experiment rather than starting a new numbered version.

## Completed Path

1. **Routing repair:** a targeted Qwen2.5-3B DPO run improved first-tool selection on customer support but did not transfer cleanly to ecommerce.
2. **Full action traces:** the benchmark was extended so agents generate tool arguments, consume tool results, answer, clarify, or abort.
3. **Executable failure benchmark:** 160 tasks now run clean and under five controlled failure types.
4. **Strict evaluation:** success requires category-appropriate behavior, required answer claims, and valid tool use.
5. **Native Qwen baseline:** established the reference behavior before repair.
6. **Recovery prompt:** tested an inference-only intervention without changing model weights.
7. **Recovery SFT:** trained on oracle recovery examples from train tasks and seen hazards; the first attempt exposed a chat-format mismatch, which was corrected and rerun.

## Current Finding

The recovery prompt is the strongest completed intervention. Corrected SFT is close to the baseline but does not improve faulted performance. This means the next work should improve the evidence and training design, not simply add another method.

## Highest-Value Next Steps

### 1. Multi-seed replication

Run the normal baseline, recovery prompt, and corrected SFT with at least two additional seeds. Report paired task-level confidence intervals. This is the main missing requirement before making a strong paper claim.

### 2. Recovery-target audit

Audit whether each training target is the right response for its failure type. A retry is sensible for a temporary server error, but a schema problem or conflicting source may require verification, fallback, or clarification. The current one-step retry-heavy data may be too narrow.

### 3. Selective reviewer

Evaluate a separate reviewer intervention only after the Qwen-only results are stable. The reviewer should be measured by both failures fixed and correct trajectories damaged. It must not be mixed into the baseline or SFT comparisons.

### 4. Preference training

Run DPO only after a validated recovery dataset and a positive or clearly interpretable SFT signal exist. DPO should use paired good and bad trajectories from the same post-error context.

### 5. External validation

If compute and API budget allow, evaluate the winning intervention on a second failure benchmark or a second model. This tests whether the result belongs to the method rather than this particular task suite and Qwen configuration.

## Evidence Gates

Do not call a repair successful unless it improves faulted performance and applied-fault recovery, preserves clean performance within three percentage points, and repeats across seeds. Always report applied-fault coverage separately from raw faulted accuracy.

## Paper Framing

The current paper should be framed as a reliability measurement and repair study. The defensible central claim is that explicit recovery instructions improve failure handling, while learned repair is sensitive to data and chat-format alignment and has not yet beaten the prompt-only intervention.

## Compute Expectation

One full native evaluation currently costs approximately 9-14 T4 GPU-hours. A two-additional-seed replication of three conditions will cost roughly 55-85 GPU-hours. DPO or reviewer studies should be scheduled only after that evidence is worth the additional compute.
