# Teaching a small model when to say "I don't know"

Can Qwen3-1.7B learn to answer SQuAD 2.0 questions from the paragraph, and to say exactly `I don't know` when the paragraph has no answer?

**Answer: yes, and supervised fine-tuning (SFT) does almost all of it.** One LoRA SFT run cut hallucinations from 51% to 12%. DPO, GRPO and twin DPO, trained on top of SFT, only moved the refusal threshold. None of them beat SFT.

## Results

Test split, 500 questions (250 answerable, 250 unanswerable), run once per selected model.

| Model | Hallucination ↓ | Over-refusal ↓ | Accuracy ↑ | Balanced ↑ |
|---|---|---|---|---|
| Qwen3-1.7B, prompt only | 50.8% | 10.8% | 71.2% | 60.2 |
| **Qwen3-1.7B + SFT** | **11.6%** | 14.0% | 81.2% | **84.8** |
| + DPO v1 | 5.6% | 24.0% | 72.0% | 83.2 |
| + DPO v2 | 12.4% | 17.2% | 76.8% | 82.2 |
| + Twin DPO (output-side) | 12.4% | 14.4% | 79.6% | 83.6 |
| + GRPO (2 runs) | dev gate failed, test not run | | | |
| Qwen3-8B, prompt only (ceiling) | 44.0% | 2.0% | 89.2% | 72.6 |

- **Hallucination:** the model answered an unanswerable question.
- **Over-refusal:** the model said "I don't know" to an answerable question.
- **Balanced:** the mean of the correct-abstain rate and the answer accuracy.

The full table, with dev numbers and AUCs, is in [`results/final/final_table.md`](results/final/final_table.md).

## What we learned

1. **SFT does the work.** Hallucinations fell from 51% to 12%, and over-refusal rose only 3 points. The tuned 1.7B beats the prompted 8B on balanced score. The 8B is more accurate, but it guesses on unanswerable questions.
2. **Nothing after SFT beat it, because the later methods only moved the threshold.** Sweeping a cut-off on SFT's own P("I don't know") traces a hallucination-vs-over-refusal curve, and every DPO, GRPO and twin-DPO model sits on it. How well P("I don't know") separates unanswerable from answerable questions (AUC 0.937 for SFT) moved by at most +0.004, with every interval including 0.
3. **Small dev gains don't survive the test set.** Twice, a dev gain of about 2 points vanished on test. Every claim here is paired and question by question, and test runs only after a dev gate.
4. **GRPO's default reward scaling makes the penalty pointless.** TRL scales advantages per group, so for two-valued groups a hallucination penalty of 1 or 3 gives the same update. A lenient answer reward also paid for padding: answers grew from 2.6 to 3.6 words.
5. **What's left is hard or noise.** Most remaining hallucinations are near-miss questions (one detail of a paragraph sentence swapped). About a fifth are SQuAD label noise, which better training can't fix.

Details and evidence: [`docs/findings.md`](docs/findings.md). Designed but not run: [`docs/future.md`](docs/future.md).

## How the experiment works

```text
SQuAD 2.0 ─► data: frozen 50/50 splits  sft 3000 · dpo 2000 · grpo 1500 · dev 300 · test 500
              │
              ├─► baseline 1.7B and ceiling 8B (prompt only)
              └─► SFT (LoRA) ─► best epoch on dev ─► the reference every later stage must beat
                     ├─► DPO v1, v2       pairs from SFT's own samples on the dpo split
                     ├─► twin DPO         pairs from a real question and its edited, unanswerable twin
                     └─► GRPO             questions where SFT's samples disagree, reward with penalties h and r

every candidate ─► dev ─► paired gate vs SFT ─► test once, only if it passes
```

The rules that keep numbers comparable:

- **One prompt, one decoding setup.** Every `metrics.json` records an `eval_fingerprint`, and runs with different fingerprints are never compared.
- **A frozen grader.** `abstention/grader.py` is pinned by a 600-case golden fixture. Changing it needs a `GRADER_VERSION` bump.
- **Paired comparisons.** `abstention/compare.py` uses a paired bootstrap and exact McNemar on the same questions. Comparing two models' separate intervals misses real differences.
- **Test is touched once per selected model.** `eval.py` refuses to overwrite a test run.

## Repo map

| Path | What |
|---|---|
| [`config.json`](config.json) | Every setting: models, prompt, splits, decoding, LoRA, per-stage hyperparameters |
| [`abstention/`](abstention/__init__.py) | The Python package; each stage is a module run with `python -m abstention.<module>` |
| [`abstention/analysis/`](abstention/analysis) | CPU-only reports over `results/`: final table, threshold curve, grader audit |
| [`tests/`](tests) | CPU tests: grader fixture, metrics, pair/pool construction, byte-identical rebuilds of committed artefacts |
| [`pod/`](pod) | Shell scripts that drive the RunPod GPU from the laptop; `pod/jobs/` holds one script per stage |
| [`results/`](results/README.md) | Every run, pulled from the pod and committed; the README explains run names |
| [`rehearsal/`](rehearsal) | Qwen3-0.6B rehearsal outputs and the split hashes the pod must reproduce |
| [`docs/`](docs) | [`findings.md`](docs/findings.md), [`running.md`](docs/running.md) (pod setup and commands), [`future.md`](docs/future.md) |

## Quick start

On a laptop (CPU only), run the tests and rebuild every report from the committed results:

```bash
uv sync
uv run python -m pytest -q
uv run python -m abstention.analysis.final_analysis              # -> results/final/
uv run --with matplotlib python -m abstention.analysis.tradeoff  # -> results/tradeoff/
uv run python -m abstention.compare sft dpo-v2 --dir results     # paired comparison of any two runs
```

Training and evaluation run on a RunPod GPU (an RTX A5000 was used). See [`docs/running.md`](docs/running.md).
