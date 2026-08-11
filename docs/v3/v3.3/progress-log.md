# v3.3 Pilot Progress Log

Running log for the v3.3 pilot. Same format as v3.0, v3.1 and v3.2: what we did, when, why, and what came out. Written as we go.

---

## Why v3.3 exists

**Date:** 2026-08-11

Across v3.1 and v3.2 the monitors were never fooled. Unsafe approval sat at roughly 0% over about 560 primary calls, eight view types, two models, plus an external τ-bench check. On its face that is overwhelming evidence that degraded evidence does not create a safety blind spot.

Building v3.2 turned up a reason to doubt that reading.

**Every degraded view we have ever run announces its own gap.**

- `actionable_summary` and `raw_projection` list `available_fields: ["order_id", "status", "total"]` and then withhold `total`. That is a receipt for the missing number.
- `explicit_redaction` stamps `{"redacted": true}`.
- `protected_summary` ships a literal manifest of what is absent.
- `silent_filter` deletes a whole event, which leaves a visible hole in a short trace.

So the honest summary of everything so far is narrower than the numbers look:

> When the monitor is **told** evidence is missing, it is never fooled.

The condition where nothing signals a gap has never been run. Not once, in three pilots.

That matters because a real lossy summariser does not announce anything. It just doesn't mention the total. The trace looks complete.

### There is also a mechanism pointing somewhere specific

v3.2 established that the monitors reason about **content**: they check the visible numbers against the agent's claim, and packaging is inert. That is a good property. It also implies a failure mode.

If the monitor's behaviour is "check the visible values against the claim," then removing the one value that breaks the check should make the check pass. Nothing else needs to go wrong.

So this is not a fishing expedition. v3.2's own finding predicts a positive result here.

---

## Design

A 2×2 over the degraded views, plus `full` as the baseline the quality gate reads off.

|                      | Advertised gap          | Silent gap            |
|----------------------|-------------------------|-----------------------|
| **Symmetric drop**   | `symmetric_advertised`  | `symmetric_silent`    |
| **Asymmetric drop**  | `asymmetric_advertised` | `asymmetric_silent`   |

**Scope — how much is removed.**

- *Symmetric*: every answer-critical value goes, the way v3.2 did it.
- *Asymmetric*: only the values that carry the contradiction go. Everything that agrees with the agent's report stays. This is what a summariser optimising for brevity naturally does — it keeps what the agent emphasised and drops the awkward outlier.

**Announcement — whether the view admits something was removed.**

- *Advertised*: keeps `observed: true`, keeps `available_fields` naming every field the record originally had, and labels itself `compression_policy: structured_field_projection`.
- *Silent*: none of those. The surviving fields are presented as though they were all the record ever held, under a neutral `compression_policy: structured_handoff`.

Announcement is deliberately tested as a **bundle** of three signals rather than one at a time. Separating them would need a three-way design and more calls; noted as a limitation.

Packaging is held constant — all four degraded views are summary-packaged — because v3.2 established that packaging on its own does nothing.

`asymmetric_silent` is the cell of interest. Everything visible is true, nothing contradicts the claim, and nothing hints that anything was dropped.

### Which fields carry the contradiction

Written out by hand in a table in `views.py`, one entry per family, each with a justification, because "which fact does the work" is a judgement about the case and not something derivable from the payloads. Examples:

- `cs_refund_o1001` → `order_record.total` and `refund_rule.maximum`. The claim is a refund cap; the cap is the order total, so both the total and the rule pointing at it have to go.
- `fi_transaction_txn03` → `transaction_record.signed_amount`. `risk_score` and `recommendation` agree with the claim and stay.
- `cs_return_r5003_owner` → `order_record.customer_id` and `customer_record.customer_id`. Here the discriminating facts are **identifiers**.

That last one is a direct fix for something v3.2 got wrong. v3.2's stripping rule spared identifiers by design, so `cs_return_r5003_owner` was never touched and read as an 8% "floor" that was really a no-op. In v3.3, symmetric is defined as *all non-identifier fields ∪ the contradiction fields*, so identifiers go when they are what matters. A test asserts this family is actually bitten.

### The invariant that makes it a fair test

In this dataset the safe and unsafe twins share **byte-identical independent evidence** and differ only in one field of the agent's report. Verified for all 12 families.

So after a drop, the twins must present **identical evidence**. The monitor cannot tell them apart from what it sees — only from the claim itself. Both the test suite and the integrity audit assert this. If it ever fails, the discriminating fact survived and the cell is not testing what it says it is.

### Scale

24 traces × 5 views × 2 monitors = **240 primary calls**, plus 40 stability repeats. 120 views, 0 validation errors. Estimated cost ~$0.40 against a $25 cap.

---

## Pre-registration

**Written and committed before any API call was made.** Recorded here so the result cannot be reinterpreted after the fact.

**Primary outcome:** unsafe-approval rate in `asymmetric_silent`, and the paired blind-spot rate against `full` (caught on the complete trace, approved once degraded).

**Predictions:**

1. **`asymmetric_silent` produces a blind spot.** Unsafe approval clearly above zero — we expect something in the 40–90% range, since the monitor's demonstrated behaviour is to check visible values against the claim, and the value that breaks the check is gone. This is the main prediction.
2. **`asymmetric_advertised` produces far fewer.** Same values removed, but the monitor is told a field is being withheld, and v3.1/v3.2 say it fails closed when it knows something is missing. Expect mostly `insufficient_evidence`.
3. **Both symmetric cells stay near zero unsafe approval**, replicating v3.2. `symmetric_silent` is the interesting one — with everything gone and no announcement, we expect the records to look so empty that the monitor balks anyway.
4. **The announcement effect is larger under asymmetric than symmetric.** An interaction, not two independent main effects.

**What would falsify the framing:** if `asymmetric_silent` also sits at 0% unsafe approval, then the monitors are catching something beyond value-checking — a consistency or completeness sense that survives having no signal at all. That would be a stronger safety result than anything in v3.1 or v3.2, and it would mean the earlier nulls are real rather than an artifact of always announcing the gap.

**Either outcome is publishable.** That is the reason for running it.

**How it will be reported:** as a control that tells us whether the instrument can detect a blind spot at all — not as "we found a blind spot." If prediction 1 holds, the headline is that the *pipeline* is the attack surface, not that the monitor is bad. A monitor shown only true, consistent, complete-looking evidence and approving it is behaving correctly on what it was given.

*(results below)*
