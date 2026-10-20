#!/usr/bin/env python3
"""Run or verify the fixed two-model CatBoost experiment without final evaluation."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for field in ("dataset_config", "dataset_dir", "artifact_dir", "budget_ledger"):
        value = Path(config[field])
        config[field] = str(value if value.is_absolute() else ROOT / value)
    config["config_path"] = str(path.resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "configs/catboost_1h_4h.json")
        if name == "verify":
            command.add_argument("--run-dir", type=Path, help="Verify this run; default is the latest completed run")
    args = parser.parse_args()
    try:
        from src.vkr_model.train import train, verify_run
        config = load_config(args.config)
        result = train(config) if args.command == "train" else verify_run(config, run_dir=args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except ImportError as error:
        parser.exit(1, "Missing training dependency; run uv sync --locked first. " + str(error) + "\n")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
