"""Teaching Qwen3-1.7B when to say "I don't know" on SQuAD 2.0.

Run any stage from the repo root as a module: `uv run python -m abstention.<module> --help`.

Shared       config     config.json loader, path helpers, eval fingerprint
             prompts    the one prompt template every stage uses
             grader     frozen grader and GRPO reward (GRADER_VERSION)
             metrics    2x2 rates, bootstrap CIs, paired comparison, McNemar, AUC
Data         data       frozen SQuAD 2.0 splits + manifest
             twins      real question / edited unanswerable twin pairs
Measure      eval       vLLM evaluation of a base model or LoRA adapter
             sample     sample a model on a split and grade every sample (DPO pairs, GRPO pool)
             probe_idk  P("I don't know") per dev question, any model
             compare    paired comparison of two runs on the same questions
Train        sft        stage 2: LoRA SFT
             dpo        stage 3: DPO variants from SFT samples
             dpo_twin   twin DPO: input-side vs output-side pairs on question twins
             grpo       stage 4: GRPO question pool, training, dev gate
Maintenance  freeze_grader  regenerate the grader's golden fixture (deliberate grader changes only)
analysis/    CPU-only reports over the tracked results/ (final table, threshold curve, hyphen audit)
"""
