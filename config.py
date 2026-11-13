"""Load config.json and resolve paths. Every script takes --config (default: config.json)."""

import hashlib
import json
import os
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "config.json"


class Config(dict):
    """The parsed config.json, plus path helpers."""

    @property
    def root(self) -> Path:
        return Path(self["paths"]["root"])

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def results_dir(self) -> Path:
        return self.path("results_dir")

    @property
    def checkpoints_dir(self) -> Path:
        return self.path("checkpoints_dir")

    def model_id(self, name_or_id: str) -> str:
        """'main' / 'debug' / 'ceiling' resolve through config.models; anything else is an HF id or a path."""
        return self["models"].get(name_or_id, name_or_id)

    def eval_fingerprint(self) -> str:
        """Hash of everything that must be identical across stages for results to be comparable:
        the prompt and the decoding settings. report.py refuses to mix rows with different fingerprints."""
        frozen = {"prompt": self["prompt"],
                  "temperature": self["eval"]["temperature"],
                  "max_new_tokens": self["eval"]["max_new_tokens"]}
        return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or DEFAULT_CONFIG)
    cfg = Config(json.loads(path.read_text()))
    hf_home = cfg["paths"].get("hf_home")
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    return cfg


def add_config_arg(parser):
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.json")
    return parser
