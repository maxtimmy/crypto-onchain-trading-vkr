#!/usr/bin/env python3
"""Train or verify fixed four-hour/24-hour controls on common 2022/2023 rows."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dataset_config", "dataset_dir", "artifact_dir", "budget_ledger"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
    config["config_path"] = str(path.resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "configs/catboost_horizons.json")
        if name == "verify":
            command.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        from src.vkr_model.horizons import run_horizons, verify_horizons
        config = load_config(args.config)
        result = run_horizons(config) if args.command == "train" else verify_horizons(config, args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except ImportError as error:
        parser.exit(1, "Missing horizon dependency; run uv sync --locked first. " + str(error) + "\n")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
