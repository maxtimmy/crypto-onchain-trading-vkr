#!/usr/bin/env python3
"""Download, validate and build the versioned BTCUSDT research dataset."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("symbol", "start", "end", "intervals", "dataset_dir")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing configuration fields: " + ", ".join(missing))
    start = datetime.fromisoformat(config["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(config["end"].replace("Z", "+00:00"))
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Use an increasing timezone-aware [start,end) range")
    if end > datetime.now(timezone.utc):
        raise ValueError("Dataset range must not include future intervals")
    if config["symbol"] != "BTCUSDT" or set(config["intervals"]) != {"1m", "1h"}:
        raise ValueError("Version 1 supports BTCUSDT with both 1m and 1h inputs")
    if config.get("feature_interval") != "1h" or config.get("horizon_hours") != 4:
        raise ValueError("Version 1 builds hourly features with a 4-hour target")
    directory = Path(config["dataset_dir"])
    config["dataset_dir"] = str(directory if directory.is_absolute() else ROOT / directory)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "validate", "build", "verify-build"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "configs/btcusdt_1h_4h.json")
        if name == "download":
            command.add_argument("--max-pages", type=int, help="Stop after a bounded number of committed pages")
        if name == "validate":
            command.add_argument("--skip-raw-hashes", action="store_true", help="Only check table coverage")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command in ("download", "validate"):
            from src.vkr_data.download import download, validate
            result = download(config, max_pages=args.max_pages) if args.command == "download" else (
                validate(config, verify_raw=not args.skip_raw_hashes))
        else:
            from src.vkr_data.build import build, verify_build
            if args.command == "build":
                from src.vkr_data.download import validate
                validation = validate(config, verify_raw=True)
                # The builder separately requires a completed ingestion snapshot.
                # Missing/flagged market bars are exclusions, not fabricated rows.
                result = build(config)
                result["input_validation"] = validation
            else:
                result = verify_build(config)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
