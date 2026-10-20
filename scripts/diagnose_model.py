#!/usr/bin/env python3
"""Diagnose saved train/validation predictions with all model fitting disabled."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for field in ("training_config", "training_run", "artifact_dir"):
        value = Path(config[field])
        config[field] = str(value if value.is_absolute() else ROOT / value)
    config["config_path"] = str(path.resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/catboost_diagnostics.json")
    args = parser.parse_args()
    try:
        from src.vkr_model.diagnose import run_diagnostics
        result = run_diagnostics(load_config(args.config))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except ImportError as error:
        parser.exit(1, "Missing diagnostic dependency; run uv sync --locked first. " + str(error) + "\n")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
