## Five saved two-value groups: normalised advantages under different penalties

**Unanswerable:** State and territorial laws preempt what kind of laws and treaties? (gold: none)

| Output | reward h=1.0,r=0.5 | reward h=3.0,r=0.5 | advantage h=1.0,r=0.5 | advantage h=3.0,r=0.5 |
|---|---|---|---|---|
| Federal law and treaties | -1.00 | -3.00 | -2.4745 | -2.4747 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |

Largest difference between the advantage columns: 1.7e-04

**Unanswerable:** What was different about the sound of historical linguistics? (gold: none)

| Output | reward h=1.0,r=0.5 | reward h=3.0,r=0.5 | advantage h=1.0,r=0.5 | advantage h=3.0,r=0.5 |
|---|---|---|---|---|
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| melody | -1.00 | -3.00 | -2.4745 | -2.4747 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | +1.00 | +1.00 | +0.3535 | +0.3535 |

Largest difference between the advantage columns: 1.7e-04

**Unanswerable:** Who attributed magical powers to Virgil in the middle ages? (gold: none)

| Output | reward h=1.0,r=0.5 | reward h=3.0,r=0.5 | advantage h=1.0,r=0.5 | advantage h=3.0,r=0.5 |
|---|---|---|---|---|
| I don't know | +1.00 | +1.00 | +0.5400 | +0.5400 |
| I don't know | +1.00 | +1.00 | +0.5400 | +0.5400 |
| I don't know | +1.00 | +1.00 | +0.5400 | +0.5400 |
| I don't know | +1.00 | +1.00 | +0.5400 | +0.5400 |
| the magical or prophetic powers | -1.00 | -3.00 | -1.6200 | -1.6201 |
| some | -1.00 | -3.00 | -1.6200 | -1.6201 |
| I don't know | +1.00 | +1.00 | +0.5400 | +0.5400 |
| I don't know | +1.00 | +1.00 | +0.5400 | +0.5400 |

Largest difference between the advantage columns: 8.7e-05

**Answerable:** Who created the concept of ecosystems? (gold: Arthur Tansley)

| Output | reward h=1.0,r=0.5 | reward h=1.0,r=1.0 | advantage h=1.0,r=0.5 | advantage h=1.0,r=1.0 |
|---|---|---|---|---|
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |
| I don't know | -0.50 | -1.00 | -2.4743 | -2.4745 |
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |
| Tansley | +0.67 | +0.67 | +0.3535 | +0.3535 |

Largest difference between the advantage columns: 1.8e-04

**Answerable:** What creates segments called a "frame"? (gold: codecs)

| Output | reward h=1.0,r=0.5 | reward h=1.0,r=1.0 | advantage h=1.0,r=0.5 | advantage h=1.0,r=1.0 |
|---|---|---|---|---|
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |
| I don't know | -0.50 | -1.00 | -2.4744 | -2.4745 |
| codecs | +1.00 | +1.00 | +0.3535 | +0.3535 |

Largest difference between the advantage columns: 1.2e-04

## Share of reward groups with 1 / 2 / 3+ distinct values (GRPO-split pool rows, 8 fresh samples each)

Training rollouts were not saved, so each run is measured at its start (SFT, independent recheck draw) and at its end (a fresh draw from the run's best checkpoint), scored with that run's own reward.

| Run | When | Class | 1 value (no signal) | 2 values (h, r cancel out) | 3+ values (h, r can matter) |
|---|---|---|---|---|---|
| G-default | start (SFT) | answerable | 24% | 46% | 30% |
| G-default | start (SFT) | unanswerable | 20% | 78% | 1% |
| G-default | start (SFT) | all | 23% | 57% | 20% |
| G-default | end (best, step 200) | answerable | 12% | 54% | 34% |
| G-default | end (best, step 200) | unanswerable | 39% | 61% | 0% |
| G-default | end (best, step 200) | all | 21% | 56% | 23% |
| G-equal | start (SFT) | answerable | 24% | 45% | 31% |
| G-equal | start (SFT) | unanswerable | 20% | 78% | 1% |
| G-equal | start (SFT) | all | 23% | 57% | 21% |
| G-equal | end (best, step 200) | answerable | 61% | 30% | 8% |
| G-equal | end (best, step 200) | unanswerable | 70% | 30% | 0% |
| G-equal | end (best, step 200) | all | 64% | 30% | 6% |
