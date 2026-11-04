"""
src/utils/config.py
--------------------
Loads config.yaml into a simple dot-accessible namespace so every script
can do:

    from src.utils.config import CFG
    print(CFG.seed)            # → 42
    print(CFG.cnn.lr)          # → 0.0003
    print(CFG.paths.results_dir)
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load config.yaml from the repo root (auto-detected) or a custom path.

    Returns
    -------
    SimpleNamespace
        Nested namespace matching the YAML structure.
    """
    if config_path is None:
        # Walk up from this file's location until we find config.yaml
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError(
                "config.yaml not found.  Run scripts from the repo root."
            )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return _dict_to_ns(raw)


# Module-level singleton — import this everywhere.
CFG = load_config()

