"""Append-only accounting of model attempts, without an experiment budget cap.

Callers coordinate writers with the shared training lock. Existing legacy
records are read as stored; no migration or rewriting of the ledger occurs.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path


STATUSES = ("STARTED", "SUCCESS", "FAILED")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty string")
    return value


def _depth(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("depth must be a positive integer")
    return int(value)


def identity(record):
    """Return the explicit ID, variant identity, or legacy run/depth identity."""
    if not isinstance(record, Mapping):
        raise ValueError("An attempt record must be a JSON object")
    if "attempt_id" in record:
        return _text(record["attempt_id"], "attempt_id")
    run_id = _text(record.get("run_id"), "run_id")
    if "variant" in record:
        return run_id, _text(record["variant"], "variant")
    return run_id, _depth(record.get("depth"))


def _validate(record):
    identity(record)
    if record.get("status") not in STATUSES:
        raise ValueError("status must be STARTED, SUCCESS or FAILED")
    for name in ("run_id", "variant", "experiment_id", "recorded_at_utc"):
        if name in record:
            _text(record[name], name)
    if "depth" in record:
        _depth(record["depth"])
    return record


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON value: " + value)


def _parse(content):
    records = []
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Attempt ledger must be UTF-8") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_object, parse_constant=_invalid_constant)
            records.append(_validate(record))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid attempt ledger record on line {}: {}".format(number, error)) from error
    return records


def load(path):
    """Read validated JSONL records; an absent or empty ledger has no records."""
    try:
        content = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    return _parse(content)


def started_count(records):
    """Count unique STARTED identities, including failed and interrupted fits."""
    started = set()
    for record in records:
        _validate(record)
        if record["status"] == "STARTED":
            started.add(identity(record))
    return len(started)


def state(path):
    """Hash and count the same bytes, without modifying the ledger."""
    try:
        content = Path(path).read_bytes()
    except FileNotFoundError:
        return {"exists": False, "sha256": None, "records": 0, "started_attempts": 0}
    records = _parse(content)
    return {"exists": True, "sha256": hashlib.sha256(content).hexdigest(),
            "records": len(records), "started_attempts": started_count(records)}


def append(path, run_id, variant, depth, status, experiment_id):
    """Append one fsynced event; the caller must hold the shared training lock.

    A valid ledger without its last newline is supported by appending a line
    separator before the new event. Existing bytes are never overwritten.
    """
    run_id = _text(run_id, "run_id")
    variant = _text(variant, "variant")
    record = {"attempt_id": run_id + "/" + variant, "run_id": run_id,
              "variant": variant, "depth": _depth(depth), "status": status,
              "experiment_id": _text(experiment_id, "experiment_id"),
              "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    _validate(record)
    path = Path(path)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        content = b""
    _parse(content)
    prefix = b"\n" if content and not content.endswith(b"\n") else b""
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(prefix + payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record
