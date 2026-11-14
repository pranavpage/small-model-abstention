# Findings

What each stage showed, with the evidence behind it. All numbers come from files in [`results/`](../results/README.md). Dev is 300 questions, test is 500, and both are 50/50 answerable/unanswerable. "Paired" means both models are compared question by question with `abstention/compare.py`.

## 1. SFT does almost all the work

| Test | Hallucination | Over-refusal | Accuracy | Balanced |
|---|---|---|---|---|
| 1.7B, prompt only | 50.8% | 10.8% | 71.2% | 60.2 |
| 1.7B + SFT | 11.6% | 14.0% | 81.2% | 84.8 |
| 8B, prompt only | 44.0% | 2.0% | 89.2% | 72.6 |

- One LoRA SFT run took 11.5 minutes on the A5000. Of the baseline's 127 test hallucinations, 102 became `I don't know`. Only 4 correct abstains became answers.
- We expected SFT to overshoot into refusing everything. It barely did: over-refusal rose 3.2 points.
- The tuned 1.7B beats the prompted 8B on balanced score. The 8B is more accurate, but it guesses on unanswerable questions.
- Read the lenient accuracy gain (+10 points), not the strict one (+16). SFT trimmed padded answers ("The Rhine is the longest river in Germany.") down to the bare span, so strict accuracy rewards a formatting fix.

## 2. Nothing after SFT beat it

| Method | Dev balanced (SFT 83.7) | Outcome |
|---|---|---|
| DPO v1 | 82.7 | "I don't know" was the preferred answer 279 times and the rejected one 172 times. Test hallucination fell 6 points, but over-refusal rose 10: every flip went the same way. |
| DPO v2 (balanced pairs) | 84.3 | Dev gain was noise; lost on test (−2.6, p = 0.011) |
| GRPO G-default (h = 1, r = 0.5, lr 1e-6) | 84.3 | Failed the dev gate (+0.7, p = 0.79) |
| GRPO G-equal (h = 1, r = 1, lr 1e-5) | 85.0 | Failed the dev gate (+1.3, p = 0.45); answers grew longer (padding) |
| Twin DPO, input-side | 85.0 | Failed the dev gate (+1.3, p = 0.29) |
| Twin DPO, output-side control | 85.7 | Passed the dev gate (+2.0, p = 0.07), then lost on test (−1.2, p = 0.24) |

The dev gate: paired p < 0.1, or a gain of at least 2 points with more questions gained than lost.

## 3. Why: they move the threshold, not the judgment

- **Every trained model sits on SFT's threshold curve.** Abstaining whenever SFT's P("I don't know") exceeds a cut-off from 0.05 to 0.95 traces a hallucination-vs-over-refusal curve ([`results/tradeoff/`](../results/tradeoff/tradeoff.md)). At its own hallucination rate, every DPO, GRPO and twin-DPO model is within noise of that curve: all paired 95% intervals include 0.
- **The ability to tell answerable from unanswerable doesn't move.** The AUC of P("I don't know"), unanswerable vs answerable, is 0.937 for SFT and 0.939 to 0.942 for every later model. Every paired interval includes 0.
- **Dev gains of about 2 points don't survive test.** It happened twice (DPO v2 and twin output-side). The dev gate and the rule of one test run per selected model exist for this reason.

## 4. What's left is mostly hard, or label noise

Hand-coded SFT mistakes ([`results/dpo-samples/analysis.md`](../results/dpo-samples/analysis.md)):

| Hallucinations | Test | Dev |
|---|---|---|
| Near-miss: the question copies a paragraph sentence with one detail swapped ("minimum" → "maximum") | 17 | 13 |
| Label noise: the paragraph does answer it | 7 | 4 |
| Vague question, loosely matching span | 4 | 2 |

- 95–99% of hallucinated answers are copied word for word from the paragraph.
- Over-refusals are often easy: the answer is stated almost word for word in a sentence mirroring the question.
- **Twin pairs had little left to teach.** SFT already gives the edited question the higher P("I don't know") on 97.5% of held-out twins. Its weak side is the real question: it prefers the gold answer on only 73% of training twins.

## 5. Method lessons

- **GRPO's default reward scaling makes the penalties pointless.** TRL 1.17.1's default `scale_rewards="group"` divides each group's advantages by that group's standard deviation. For a group with two reward values, h = 1 and h = 3 give advantages equal to within 2e-4 ([`results/final/reward_groups.md`](../results/final/reward_groups.md)). Only about 20% of groups have three or more values, so G-equal vs G-default mainly tested the learning rate.
- **A lenient answer reward pays for padding.** Any answer of up to 20 words that contains the gold scores 1. G-equal's dev answers grew from 2.6 to 3.6 words, and the lenient − strict gap from 0.033 to 0.084 ([`results/final/padding.md`](../results/final/padding.md)). One output listed candidate spans until one matched.
- **Input-side DPO has no anchor.** Its loss only compares the same response after two questions, so the overall level of "I don't know" can drift. At lr 2e-5 it fell on every question, and dev AUC dropped from 0.937 to 0.909.
- **GRPO needs disagreeing groups.** SFT gets most questions right on every sample, so only 518 of 1,500 GRPO-split questions produce a non-zero advantage. Answerable questions whose only contrast is "I don't know" vs a wrong answer teach guessing, so they were dropped.
- **Known grader limitation: hyphens.** The grader deletes `-`, so "ADSL-broadband" fails to match "ADSL". It misgrades 17 of 4,950 dev answers and 2 of 1,750 test answers across every run. No ranking reverses and no gate decision changes, so the grader stays frozen ([`results/grader-audit/hyphens.md`](../results/grader-audit/hyphens.md)).
- **Paired tests beat interval overlap.** On test, DPO v2's balanced-score interval overlaps SFT's ([78.8, 85.4] vs [81.6, 87.8]), yet the paired comparison shows a real loss: −2.6 [−4.6, −0.8], p = 0.011.

## 6. Bottom line

For a 1.7B model on this task, SFT takes balanced score from 60 to 85. The preference and RL methods we tried only picked a different point on SFT's own trade-off curve. [`future.md`](future.md) has two follow-ups designed but not run:

1. GRPO with batch-level or no reward scaling and an exact-match answer reward.
2. Twin DPO on hand-checked hard twins.
