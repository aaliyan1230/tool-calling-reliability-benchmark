"""v3.3: does the monitor stay safe when nothing tells it evidence is missing?

Across v3.1 and v3.2 the monitors were never fooled — unsafe approval sat at
roughly 0% over ~560 primary calls and eight view types. But every one of those
degraded views **announced its own gap**: `available_fields` listed the withheld
field by name, `explicit_redaction` stamped `{"redacted": true}`, and
`protected_summary` shipped a manifest of what was absent.

So the honest reading of v3.1 and v3.2 is narrower than the numbers suggest:

    when the monitor is told evidence is missing, it is never fooled.

The condition where nothing signals a gap has never been run. v3.3 runs it.

Two factors:

                      advertised gap        silent gap
    symmetric drop    symmetric_advertised  symmetric_silent
    asymmetric drop   asymmetric_advertised asymmetric_silent

- **symmetric** removes every answer-critical value, the way v3.2 did.
- **asymmetric** removes only the values that carry the contradiction, leaving
  everything that agrees with the agent's report. This is what a summariser
  optimising for brevity naturally does: it keeps what the agent emphasised.
- **advertised** keeps the gap signals. **silent** removes them, so the record
  looks like it simply never held the missing field.

`asymmetric_silent` is the cell of interest: everything visible is true, nothing
contradicts the claim, and nothing hints that anything was dropped.

Packaging is held constant (all four degraded views are summary-packaged),
because v3.2 established that packaging on its own does nothing.
"""
