# Parked experiments

Designed but not run. Every number below comes from results already in `results/`; nothing here needed a GPU.

## 1. GRPO with batch-level or no reward scaling, and an exact-match answer reward

**Why.** The two GRPO runs had two problems:

- **The penalties didn't count.** TRL 1.17.1's default `scale_rewards="group"` divides each group's advantages by the group's own standard deviation. A group with two reward values then gets the same advantages whatever their size, so h (the hallucination penalty) makes no difference (`results/final/reward_groups.md`).
- **Padding paid.** The lenient answer score gives full credit to any answer of ≤ 20 words that contains a gold answer. G-equal (lr 1e-5) got longer as it trained. Mean dev answer length went from 2.64 to 3.64 words by step 150, the lenient − strict gap went from 0.033 to 0.084 by step 250, and the run was stopped at step 250.

**Design.**

- **Scaling:** two runs that differ only in `scale_rewards`: `"batch"` (divide by the standard deviation of the whole generation batch) and `"none"` (Dr. GRPO: advantage = reward − group mean). Both are supported values of TRL's `GRPOConfig.scale_rewards`; check against the pod's TRL version in the dry run.
- **Answer reward:** `exact` instead of lenient. 1 if the normalised output equals a normalised gold answer, else 0. Abstain (−r), hallucination (−h) and the format penalty are unchanged.
- **Settings:** h = 1, r = 0.5, lr 1e-5 (G-equal's rate; G-default's 1e-6 barely moved). Same KL, steps, gate and evaluation as G-default.
- **Evaluation grader:** unchanged, so dev/test numbers stay comparable with every earlier model.
- **Pool:** rebuild it with the exact-match reward. Under the lenient reward a padded answer and an exact one both score 1, so the groups where padding and exact answers differ never enter the current pool. "At least one correct sample" for answerable questions should mean at least one exact match.

**What's known already.**

- **SFT is often correct but not exact.** On dev, 94 of SFT's 120 correct answers are exact matches (177 of 203 on test). Under this reward about a fifth of its correct answers would score 0, so the run first has to teach exact spans. Note that SQuAD train questions have a single gold answer.
- **Fixture impact.** On the 600-case grader fixture, the exact-match reward differs from the lenient one (h = 1, r = 0.5) on 182 cases. Adding the setting leaves every existing graded case and reward identical, the same precedent as G-equal: no version bump.
- **The reward still prefers guessing.** On an answerable question, a wrong answer still scores 0 and "I don't know" scores −0.5.

**Cost.** About 40 min of training per run plus dev evals (G-default: 342 steps in 2,372 s on the A5000).

**Code.** A patch with the reward option and the extended fixture exists in the workspace's gitignored `.context/` (not in the repo). It still needs the `abstention/grpo.py` side: per-run `answer_reward`, `scale_rewards` and pool filter.

## 2. Twin DPO on hand-checked hard twins

**Why.** Twin DPO (PR #8) didn't help because SFT already tells most twins apart. On held-out twins, it gives the edited question the higher P("I don't know") 97.5% of the time. The twins it misses (P(IDK | edited) ≤ 0.5 on 27 of 200 held-out twins) mix two kinds:

- real near-misses: "minimum" → "maximum", "son" → "dad";
- SQuAD edits that don't remove the answer: "Houston's" → "Texas's major newspaper", "sail around" → "sail through".

**Design.**

- Score all ~11.9k one-to-one twins in SQuAD train with SFT and keep the ones where it doesn't abstain on the edited question (~13% if the held-out rate holds, about 1,500).
- Read each with its paragraph and keep it only if the edit really removes the answer.
- Train on the combined input-side + output-side loss, so the overall level of "I don't know" can't drift. At lr 2e-5 the input-side-only loss lowered P(IDK) on every question and lost AUC: 0.937 → 0.909 on dev.

**Ceiling.** Near-miss questions are 13 of SFT's 19 dev hallucinations; label noise is 4 of 19 dev and 7 of 29 test, and better training data can't fix those. A realistic gain is a few dev questions.
