"""The one prompt template used by every stage (eval, SFT targets, DPO sampling, GRPO rollouts)."""


def build_messages(cfg, row) -> list[dict]:
    p = cfg["prompt"]
    user = p["user_template"].format(context=row["context"], question=row["question"])
    return [{"role": "system", "content": p["system"]}, {"role": "user", "content": user}]


def render(cfg, tokenizer, row) -> str:
    """Chat-formatted prompt string ending at the assistant turn."""
    kwargs = {}
    if "enable_thinking" in (tokenizer.chat_template or ""):
        kwargs["enable_thinking"] = cfg["prompt"]["enable_thinking"]
    return tokenizer.apply_chat_template(
        build_messages(cfg, row), tokenize=False, add_generation_prompt=True, **kwargs)
