# Draft message to Monika

Hi Monika,

I’ve been running a small pilot on evidence provenance in agent monitoring. We built 16 matched simulated cases across four tool-use domains, varying whether an output was correct or corrupted and whether the tool called itself “verified” or warned it might be unreliable.

The label effect was model-specific: 9.4 points for DeepSeek V4 Flash and 3.1 for GPT-5.6 Terra, with wide intervals. Our pre-set gate for a broad label-trust effect did not pass. The more consistent result was that a short evidence-first instruction cut false alarms from 15.6% to 3.1% on DeepSeek and 9.4% to 0% on GPT, while corruption detection stayed at 100%. It also improved repeatability on a four-case rerun.

I’d really value your take on two questions: Is provenance-sensitive monitoring useful to pursue beyond prompting, and what realistic setting would make the strongest next test?

Happy to send the one-page brief and exact trajectories.
