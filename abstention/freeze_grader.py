"""Snapshot the grader's behaviour on 600 real model outputs into a golden fixture.

test_grader.py replays the fixture and fails on any difference, so the grader can't change silently.
Only re-run this after a deliberate grader change, and bump GRADER_VERSION in grader.py first.
The fixture is self-contained: it re-grades the cases already stored in it.

Usage: uv run python -m abstention.freeze_grader
"""

import json

from abstention.config import REPO_ROOT
from abstention.grader import GRADER_VERSION, grade, reward

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "grader_golden.jsonl"
REWARD_SETTINGS = [(1.0, 0.5), (0.3, 0.5), (3.0, 0.5), (3.0, 0.0), (1.0, 1.0)]  # G-default, G-low, G-high, G-hack, G-equal
CASE_KEYS = ["source", "id", "output", "answerable", "gold"]


def expected(output, answerable, gold):
    return {
        "grade": grade(output, answerable, gold),
        "rewards": {f"h={h},r={r}": reward(output, answerable, gold, h, r)["reward"] for h, r in REWARD_SETTINGS},
    }


def main():
    cases = [json.loads(l) for l in FIXTURE.read_text().splitlines()[1:]]
    with open(FIXTURE, "w") as f:
        f.write(json.dumps({"grader_version": GRADER_VERSION}) + "\n")
        for c in cases:
            case = {k: c[k] for k in CASE_KEYS}
            f.write(json.dumps({**case, **expected(c["output"], c["answerable"], c["gold"])}) + "\n")
    print(f"froze grader v{GRADER_VERSION} on {len(cases)} outputs -> {FIXTURE}")


if __name__ == "__main__":
    main()
