# Known grader limitation: hyphens and dashes (grader v1.0, unchanged)

The grader deletes `-` without a space and keeps en/em dashes, so hyphenated words can fail to match. Every saved eval output was regraded with dashes treated as spaces; only answerable outputs can change.

| Split | Answerable outputs | Score changed | Cell flipped | Misgraded by v1.0 |
|---|---|---|---|---|
| dev | 4950 | 90 | 17 | 17 |
| test | 1750 | 25 | 4 | 2 |

| Gold | Output | v1.0 | Dashes as spaces | v1.0 wrong? | Runs |
|---|---|---|---|---|---|
| 512-bit | 1024-bit | wrong (0.00) | correct (0.50) | no: different number; dashes-as-spaces wrongly credits the shared word 'bit' | baseline |
| mid-2000s | 2000s | wrong (0.00) | correct (0.67) | no: drops 'mid' (the decade, not its middle); v1.0 'wrong' is defensible | baseline |
| ADSL | ADSL-broadband service is provided on the island. | wrong (0.00) | correct (1.00) | yes: right content in a sentence (the format rule, not the 2x2, covers sentences) | baseline-0.6b-dev, ceiling-dev |
| Hargeisa | Hargeisa-based | wrong (0.00) | correct (1.00) | yes: gold span plus a hyphenated suffix | baseline-dev, grpo-G-equal-dev-s150, grpo-G-equal-dev-s200, grpo-G-equal-dev-s250 |
| four | four-valve | wrong (0.00) | correct (1.00) | yes: gold number inside a hyphenated compound | ceiling |
| ten | 10 horsepower | wrong (0.00) | correct (0.50) | yes: right answer (10 = ten); v1.0 misses it because 'ten-horsepower' becomes one token, and dashes-as-spaces credits it only through the word 'horsepower' | ceiling |
| Hargeisa | Hargeisa-based telecommunications firm | wrong (0.00) | correct (1.00) | yes: gold span plus a hyphenated suffix | ceiling-dev |
| ADSL | ADSL-broadband | wrong (0.00) | correct (1.00) | yes: gold span plus a hyphenated qualifier | dpo-twin-out-dev-lr2e-05-e1, dpo-twin-out-dev-lr2e-05-e2, dpo-v2-dev-lr2e-05-e1, dpo-v2-dev-lr2e-05-e2, dpo-v2-dev-lr5e-06-e1, dpo-v2-dev-lr5e-06-e2, grpo-G-default-dev-s100, grpo-G-default-dev-s50, sft-dev-e1, sft-dev-e2 |

## Does correcting the misgrades change any ranking?

| Model | Split | Misgraded | Balanced (v1.0) | Balanced (misgrades corrected) |
|---|---|---|---|---|
| Baseline 1.7B | dev | 1 | 59.33 | 59.67 |
| SFT | dev | 1 | 83.67 | 84.00 |
| DPO v1 | dev | 0 | 82.67 | 82.67 |
| DPO v2 | dev | 1 | 84.33 | 84.67 |
| G-default best (step 200) | dev | 0 | 84.33 | 84.33 |
| G-equal best (step 200) | dev | 1 | 85.00 | 85.33 |
| Twin DPO input-side (lr5e-06 e2) | dev | 0 | 85.00 | 85.00 |
| Twin DPO output-side (lr5e-06 e2) | dev | 0 | 85.67 | 85.67 |
| 8B ceiling | dev | 2 | 76.67 | 77.33 |
| Baseline 1.7B | test | 0 | 60.20 | 60.20 |
| SFT | test | 0 | 84.80 | 84.80 |
| DPO v1 | test | 0 | 83.20 | 83.20 |
| DPO v2 | test | 0 | 82.20 | 82.20 |
| Twin DPO output-side (lr5e-06 e2) | test | 0 | 83.60 | 83.60 |
| 8B ceiling | test | 2 | 72.60 | 73.00 |

dev: no strict reversal; ties broken: DPO v2 now above G-default best (step 200); G-equal best (step 200) now above Twin DPO input-side (lr5e-06 e2)

test: no strict reversal

Runs with a flip: baseline (2), baseline-0.6b-dev (1), baseline-dev (1), ceiling (2), ceiling-dev (2), dpo-twin-out-dev-lr2e-05-e1 (1), dpo-twin-out-dev-lr2e-05-e2 (1), dpo-v2-dev-lr2e-05-e1 (1), dpo-v2-dev-lr2e-05-e2 (1), dpo-v2-dev-lr5e-06-e1 (1), dpo-v2-dev-lr5e-06-e2 (1), grpo-G-default-dev-s100 (1), grpo-G-default-dev-s50 (1), grpo-G-equal-dev-s150 (1), grpo-G-equal-dev-s200 (1), grpo-G-equal-dev-s250 (1), sft-dev-e1 (1), sft-dev-e2 (1)

Same-question flips cancel in paired comparisons between models that both made them.
