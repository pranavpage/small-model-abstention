"""P("I don't know<|im_end|>" | prompt) on every dev question, for several models (runs on the RunPod GPU, no training).

Same probe as dpo.IdkProbe / the GRPO monitor (exact eval prompt, full-sequence probability under teacher forcing).
Writes results/idk-probe/<label>.json: {"model", "adapter", "p_idk": {question id: P}}.

Usage (on the pod):
  python probe_idk.py SFT=main:checkpoints/sft/best Baseline=main Ceiling=ceiling ...
"""

import argparse
import json
import time

from config import add_config_arg, load_config


def main():
    p = add_config_arg(argparse.ArgumentParser())
    p.add_argument("specs", nargs="+", help="LABEL=MODEL[:ADAPTER]")
    p.add_argument("--split", default="dev")
    args = p.parse_args()
    cfg = load_config(args.config)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dpo import IdkProbe

    rows = [json.loads(l) for l in open(cfg.data_dir / f"{args.split}.jsonl")]
    out = cfg.results_dir / "idk-probe"
    out.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for spec in args.specs:
        label, rest = spec.split("=", 1)
        model_name, _, adapter = rest.partition(":")
        model_id = cfg.model_id(model_name)
        t0 = time.time()
        if model_id not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            tok = AutoTokenizer.from_pretrained(model_id)
            base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda")
            loaded[model_id] = (tok, base, IdkProbe(cfg, tok, rows))
        tok, base, probe = loaded[model_id]
        if adapter:
            model = PeftModel.from_pretrained(base, cfg.root / adapter)
            stats = probe(model)
            base = model.unload()  # back to the bare base model for the next spec
            loaded[model_id] = (tok, base, probe)
        else:
            stats = probe(base)
        res = {"label": label, "model": model_id, "adapter": adapter or None, "split": args.split, "stats": stats,
               "p_idk": {r["id"]: q for r, q in zip(rows, probe.last_p)}}
        json.dump(res, open(out / f"{label}.json", "w"), indent=1)
        print(f"{label}: P(IDK) answerable {stats['answerable']['mean_p']:.3f} unanswerable "
              f"{stats['unanswerable']['mean_p']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
