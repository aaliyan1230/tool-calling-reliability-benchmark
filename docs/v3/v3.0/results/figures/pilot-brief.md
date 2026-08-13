# Pilot brief: Do monitors trust a tool’s confidence label?

## Question

When a tool calls its own answer `verified`, does an AI monitor trust that label over independent records?

## Test

We built 16 realistic simulated cases across customer support, ecommerce, fintech, and developer tools. Each case was shown in four matched forms: correct/corrupt output × verified/warning label. Only one factor changed at a time. We compared a general baseline prompt with an evidence-first prompt that explicitly prioritizes independent records.

## Results

| Model | Prompt | Self-label effect | False alarms | Corruption detected | Accuracy |
|---|---|---:|---:|---:|---:|
| DeepSeek V4 Flash | Baseline | 9.4 pp | 15.6% | 100.0% | 92.2% |
| DeepSeek V4 Flash | Evidence-first | 3.1 pp | 3.1% | 100.0% | 98.4% |
| GPT-5.6 Terra | Baseline | 3.1 pp | 9.4% | 100.0% | 95.3% |
| GPT-5.6 Terra | Evidence-first | 0.0 pp | 0.0% | 100.0% | 100.0% |

## What this means

For DeepSeek, the evidence-first prompt reduced the measured label effect by 66.7% and false alarms by 12.5 percentage points, while corruption detection stayed unchanged.
For GPT-5.6 Terra, evidence-first reduced false alarms by 9.4 percentage points and improved accuracy by 4.7 percentage points; corruption detection again stayed at 100%.
On the four-case repeat check, decision unanimity changed from 87.5% to 100.0% for DeepSeek and from 81.2% to 93.8% for GPT.

## Limits

The traces are simulated and the 16-case sample is small. Both baseline label-effect confidence intervals include zero. Two developer-tool cases also showed that `allow` can be confused with approving a failed build rather than accepting a correct status report.

## Next decision

The pre-registered contact gate did not pass. Share this as an honest early result about model-specific label sensitivity and a promising evidence-first mitigation—not as proof of a broad failure mode.
