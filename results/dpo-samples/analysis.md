# Is DPO worth running? SFT error analysis + DPO pair yield (analysis only, no training)

Model: Qwen3-1.7B + `checkpoints/sft/best` (epoch 2). Grader v1.0, eval fingerprint d2bf52ad8dfb.

## SFT mistakes by type (hand-coded)

All test (`sft`) and dev (`sft-dev-e2`) mistakes, plus a random sample of greedy mistakes on the DPO split.

| Type | Test | Dev | DPO split (sample) |
|---|---|---|---|
| **Hallucination** (answered an unanswerable question) | 29 | 19 | 25 of 97 |
| near-miss: question copies a paragraph sentence with one detail swapped (entity, number, negation, verb); model answers from that sentence | 17 | 13 | 19 |
| label noise: the paragraph does answer it | 7 | 4 | 5 |
| vague question, loosely matching span | 4 | 2 | 1 |
| outside knowledge | 1 | 0 | 0 |
| **Over-refusal** (said "I don't know" to an answerable question) | 35 | 18 | 25 of 140 |
| easy: answer stated almost verbatim in a sentence mirroring the question | 13 | 10 | 11 |
| paraphrase / one-step inference needed | 9 | 6 | 10 |
| long clause answer (why / what happened) | 5 | 1 | 1 |
| ill-posed: yes/no, false premise, bad gold | 8 | 1 | 3 |
| **Wrong answer** | 12 | 12 | 12 of 67 |
| acceptable or partial span the grader scores < 0.5 | 4 | 7 | 7 |
| wrong span (neighbouring entity) or outside knowledge | 8 | 5 | 5 |

Automatic features (same code on all three; greedy decoding):

| | Test | Dev | DPO split |
|---|---|---|---|
| hallucination rate | 11.6% | 12.7% | 9.7% |
| over-refusal rate | 14.0% | 12.0% | 14.0% |
| wrong rate | 4.8% | 8.0% | 6.7% |
| hallucinated output copied verbatim from the paragraph | 97% | 95% | 99% |
| over-refused gold answer, median words / share > 5 words | 3 / 23% | 3 / 17% | 2 / 20% |

## DPO pair yield (`pair_counts.json`)

2,000 DPO-split questions, 4 samples each at T=1.0 (+1 greedy). 52 s of generation on the A5000.

| | Answerable | Unanswerable | Total |
|---|---|---|---|
| questions with >= 1 pair | 383 (over-refusal rejected: 264, wrong: 175) | 192 | **575** |
| ... where the model also sampled the good output (on-policy chosen) | 263 | 129 | 392 |
| unique (chosen, rejected) pairs | 706 | 279 | 985 |
| greedy mistakes covered by a pair | 205 / 207 | 97 / 97 | |
| questions with >= 1 pair after dropping partial-span "wrong" rejecteds | 347 | 192 | 539 |
