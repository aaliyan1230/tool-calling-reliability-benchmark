"""v3.2: does the monitor fail closed because of missing content or because of summary packaging?

v3.1 established that monitors stop approving anything when evidence is degraded.
It could not say why, because every degraded view changed two things at once: it
removed answer-critical values *and* it replaced raw events with a summary.

v3.2 separates those two factors into a 2x2:

                       values present        values stripped
    raw packaging      full                  raw_projection
    summary packaging  lossless_summary      actionable_summary

`full` and `actionable_summary` are carried over unchanged from v3.1, so this run
also replicates the v3.1 numbers. The two new views are the corners that were
missing.

Everything else — the system prompt, the models, the sampling settings, the
deterministic scoring — is imported from v3.1 rather than restated, so the only
thing that differs between the runs is the view set.
"""
