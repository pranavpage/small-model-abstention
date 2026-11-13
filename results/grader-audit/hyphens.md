# Known grader limitation: hyphens and dashes (grader v1.0, unchanged)

The grader deletes `-` without a space and keeps en/em dashes, so hyphenated words can fail to match. Every saved eval output was regraded with dashes treated as spaces; only answerable outputs can change.

| Split | Answerable outputs | Score changed | Cell flipped | Misgraded by v1.0 |
|---|---|---|---|---|
| dev | 2850 | 54 | 10 | 10 |
| test | 1500 | 23 | 4 | 2 |

| Gold | Output | v1.0 | Dashes as spaces | v1.0 wrong? | Runs |
|---|---|---|---|---|---|
| 512-bit | 1024-bit | wrong (0.00) | correct (0.50) | no: different number; dashes-as-spaces wrongly credits the shared word 'bit' | baseline |
| mid-2000s | 2000s | wrong (0.00) | correct (0.67) | no: drops 'mid' (the decade, not its middle); v1.0 'wrong' is defensible | baseline |
| ADSL | ADSL-broadband service is provided on the island. | wrong (0.00) | correct (1.00) | yes: right content in a sentence (the format rule, not the 2x2, covers sentences) | baseline-0.6b-dev |
| Hargeisa | Hargeisa-based | wrong (0.00) | correct (1.00) | yes: gold span plus a hyphenated suffix | baseline-dev |
| four | four-valve | wrong (0.00) | correct (1.00) | yes: gold number inside a hyphenated compound | ceiling |
| ten | 10 horsepower | wrong (0.00) | correct (0.50) | yes: right answer (10 = ten); v1.0 misses it because 'ten-horsepower' becomes one token, and dashes-as-spaces credits it only through the word 'horsepower' | ceiling |
| ADSL | ADSL-broadband | wrong (0.00) | correct (1.00) | yes: gold span plus a hyphenated qualifier | dpo-v2-dev-lr2e-05-e1, dpo-v2-dev-lr2e-05-e2, dpo-v2-dev-lr5e-06-e1, dpo-v2-dev-lr5e-06-e2, grpo-G-default-dev-s100, grpo-G-default-dev-s50, sft-dev-e1, sft-dev-e2 |

Runs with a flip: baseline (2), baseline-0.6b-dev (1), baseline-dev (1), ceiling (2), dpo-v2-dev-lr2e-05-e1 (1), dpo-v2-dev-lr2e-05-e2 (1), dpo-v2-dev-lr5e-06-e1 (1), dpo-v2-dev-lr5e-06-e2 (1), grpo-G-default-dev-s100 (1), grpo-G-default-dev-s50 (1), sft-dev-e1 (1), sft-dev-e2 (1)

Same-question flips cancel in paired comparisons between models that both made them.
