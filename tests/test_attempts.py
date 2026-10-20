"""Stable attempt identities and append-only compatibility with legacy JSONL."""

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.vkr_model.attempts import append, identity, load, started_count, state


class AttemptsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "attempts.jsonl"

    def write(self, records):
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def test_mixed_two_legacy_and_three_same_depth_variants_count_five(self):
        records = [{"run_id": "legacy", "depth": depth, "status": "STARTED"} for depth in (4, 6)]
        records += [{"attempt_id": "controls/" + variant, "run_id": "controls", "variant": variant,
                     "depth": 4, "status": "STARTED"} for variant in ("constant", "shallow", "regularized")]
        self.write(records)
        before = self.path.read_bytes()
        self.assertEqual(load(self.path), records)
        self.assertEqual(started_count(records), 5)
        self.assertEqual(state(self.path)["started_attempts"], 5)
        self.assertEqual(self.path.read_bytes(), before)

    def test_identity_precedence_and_hashability(self):
        self.assertEqual(identity({"attempt_id": "explicit", "run_id": "run", "variant": "x", "depth": 4}), "explicit")
        self.assertEqual(identity({"run_id": "run", "variant": "x", "depth": 4}), ("run", "x"))
        self.assertEqual(identity({"run_id": "run", "depth": 4}), ("run", 4))
        self.assertEqual(len({identity({"run_id": "run", "depth": 4}), identity({"run_id": "run", "depth": 4})}), 1)
        self.assertNotEqual(identity({"run_id": "run", "variant": "a", "depth": 4}),
                            identity({"run_id": "run", "variant": "b", "depth": 4}))

    def test_repeated_started_and_failed_events_do_not_decrement_or_duplicate(self):
        started = {"run_id": "run", "variant": "a", "depth": 4, "status": "STARTED"}
        failed = dict(started, status="FAILED")
        records = [started, dict(started), failed, dict(failed)]
        self.assertEqual(started_count(records), 1)
        self.assertEqual(started_count([failed]), 0)
        self.assertEqual(started_count(records + [dict(started, variant="b", status="SUCCESS")]), 1)

    def test_interrupted_started_event_remains_an_attempt(self):
        self.write([{"run_id": "interrupted", "depth": 4, "status": "STARTED"}])
        self.assertEqual(state(self.path)["started_attempts"], 1)

    def test_accounting_has_no_twenty_attempt_cap(self):
        records = [{"attempt_id": "run/" + str(index), "status": "STARTED"} for index in range(25)]
        self.write(records)
        self.assertEqual(started_count(load(self.path)), 25)
        self.assertEqual(state(self.path)["started_attempts"], 25)

    def test_missing_empty_and_whitespace_ledgers(self):
        self.assertEqual(load(self.path), [])
        self.assertEqual(state(self.path), {"exists": False, "sha256": None, "records": 0, "started_attempts": 0})
        self.assertFalse(self.path.exists())
        for content in (b"", b"\n \n\t"):
            self.path.write_bytes(content)
            self.assertEqual(load(self.path), [])
            self.assertEqual(state(self.path), {"exists": True, "sha256": hashlib.sha256(content).hexdigest(),
                                              "records": 0, "started_attempts": 0})

    def test_state_hash_and_record_count_describe_exact_original_bytes(self):
        content = b'\n {"run_id":"old", "depth":4,"status":"STARTED"}\n\n'
        self.path.write_bytes(content)
        self.assertEqual(state(self.path), {"exists": True, "sha256": hashlib.sha256(content).hexdigest(),
                                          "records": 1, "started_attempts": 1})
        self.assertEqual(self.path.read_bytes(), content)

    def test_append_retains_legacy_prefix_and_records_stable_id_utc_and_fsync(self):
        old = b'{"run_id":"old","depth":4,"status":"STARTED"}'
        self.path.write_bytes(old)  # no trailing newline
        with patch("src.vkr_model.attempts.os.fsync", wraps=__import__("os").fsync) as fsync:
            record = append(self.path, "controls", "shallow", 4, "STARTED", "0005")
        self.assertEqual(fsync.call_count, 1)
        self.assertTrue(self.path.read_bytes().startswith(old + b"\n"))
        self.assertEqual(record["attempt_id"], "controls/shallow")
        self.assertEqual(record["run_id"], "controls")
        self.assertEqual(record["variant"], "shallow")
        self.assertEqual(record["experiment_id"], "0005")
        self.assertTrue(record["recorded_at_utc"].endswith("Z"))
        self.assertEqual(datetime.fromisoformat(record["recorded_at_utc"].replace("Z", "+00:00")).tzinfo, timezone.utc)
        append(self.path, "controls", "shallow", 4, "FAILED", "0005")
        records = load(self.path)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1], record)
        self.assertEqual(identity(records[1]), identity(records[2]))
        self.assertEqual(started_count(records), 2)

    def test_append_creates_parent_and_an_absent_ledger(self):
        path = self.path.parent / "nested" / "events.jsonl"
        record = append(path, "run", "x", 4, "STARTED", "0005")
        self.assertEqual(load(path), [record])

    def test_malformed_ledger_is_rejected_instead_of_silently_dropping_events(self):
        invalid = [b'{"run_id":"cut', b'[]\n', b'null\n', b'""\n', b'{}\n', b'\xff',
                   b'{"run_id":"r","depth":4,"status":"STARTED","status":"FAILED"}\n',
                   b'{"run_id":"r","depth":NaN,"status":"STARTED"}\n']
        for content in invalid:
            self.path.write_bytes(content)
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    load(self.path)
                with self.assertRaises(ValueError):
                    state(self.path)
                with self.assertRaises(ValueError):
                    append(self.path, "new", "x", 4, "STARTED", "0005")
                self.assertEqual(self.path.read_bytes(), content)

    def test_empty_id_invalid_depth_and_unknown_status_are_rejected(self):
        invalid = [
            {"attempt_id": "", "run_id": "r", "depth": 4, "status": "STARTED"},
            {"attempt_id": 4, "status": "STARTED"},
            {"run_id": "r", "variant": " ", "depth": 4, "status": "STARTED"},
            {"run_id": "", "depth": 4, "status": "STARTED"},
            {"run_id": "r", "depth": True, "status": "STARTED"},
            {"run_id": "r", "depth": 4.0, "status": "STARTED"},
            {"run_id": "r", "depth": 0, "status": "STARTED"},
            {"run_id": "r", "depth": 4, "status": "OTHER"},
            {"run_id": "r", "depth": 4},
        ]
        for record in invalid:
            self.write([record])
            with self.subTest(record=record), self.assertRaises(ValueError):
                load(self.path)

    def test_append_validates_arguments_before_creating_file(self):
        invalid = [("", "x", 4, "STARTED", "0005"), ("r", "", 4, "STARTED", "0005"),
                   ("r", "x", False, "STARTED", "0005"), ("r", "x", 4, "OTHER", "0005"),
                   ("r", "x", 4, "STARTED", "")]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                append(self.path, *args)
            self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
