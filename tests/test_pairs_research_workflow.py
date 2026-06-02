import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools import multiple_comparisons, pairs_candidate_campaign
from walkforward_pairs import _slice_pair_window, _tag_symbol


class TestPairsResearchWorkflow(unittest.TestCase):
    def test_slice_pair_window_preserves_leg_symbols(self):
        df_y = _tag_symbol(pd.DataFrame({
            "timestamp": ["2026-01-02T14:30:00Z", "2026-01-03T14:30:00Z"],
            "open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [10, 10],
        }), "JPM")
        df_x = _tag_symbol(pd.DataFrame({
            "timestamp": ["2026-01-02T14:30:00Z", "2026-01-03T14:30:00Z"],
            "open": [3, 4], "high": [3, 4], "low": [3, 4], "close": [3, 4], "volume": [10, 10],
        }), "BAC")
        sliced = _slice_pair_window(
            df_y,
            df_x,
            pd.Timestamp("2026-01-02", tz="UTC"),
            pd.Timestamp("2026-01-02", tz="UTC"),
        )
        self.assertEqual(sorted(sliced["symbol"].tolist()), ["BAC", "JPM"])

    def test_multiple_comparisons_honors_preregistered_family_size(self):
        tests = [
            multiple_comparisons.TestResult(pair="A/B", raw_p=0.012),
            multiple_comparisons.TestResult(pair="C/D", raw_p=0.03),
        ]
        results = multiple_comparisons.apply_corrections(tests, alpha=0.05, family_size=5)
        self.assertFalse(results[0].bonferroni_pass)
        self.assertAlmostEqual(results[0].bonferroni_threshold, 0.01)

    def test_classify_candidate_buckets(self):
        gates = {
            "a2": pairs_candidate_campaign.GateResult("PASS", ""),
            "a3": pairs_candidate_campaign.GateResult("PASS", ""),
            "a4": pairs_candidate_campaign.GateResult("PASS", ""),
            "d3": pairs_candidate_campaign.GateResult("PASS", ""),
        }
        self.assertEqual(
            pairs_candidate_campaign.classify_candidate(True, True, gates),
            "APPROVED",
        )
        gates["d3"] = pairs_candidate_campaign.GateResult("ATTENTION", "")
        self.assertEqual(
            pairs_candidate_campaign.classify_candidate(True, True, gates),
            "PROBATION",
        )
        gates["d3"] = pairs_candidate_campaign.GateResult("FAIL", "")
        self.assertEqual(
            pairs_candidate_campaign.classify_candidate(True, True, gates),
            "REJECTED",
        )

    def test_walkforward_summary_uses_ci_and_pvalue(self):
        payload = {
            "results": [{
                "symbol": "JPM/BAC",
                "results": [
                    {"test_result": {"total_return_pct": 0.002}},
                    {"test_result": {"total_return_pct": 0.003}},
                    {"test_result": {"total_return_pct": 0.004}},
                ],
                "total_test_pnl": 123.0,
            }]
        }
        summary = pairs_candidate_campaign.summarize_walkforward_payload(payload, bootstrap=1000, alpha=0.05)
        self.assertEqual(summary["windows"], 3)
        self.assertGreater(summary["ci_lo"], 0.0)
        self.assertTrue(math.isfinite(summary["raw_p"]))


class TestRunCommandStreaming(unittest.TestCase):
    """The campaign sub-step runner — live streaming + tee-to-log (the fix for
    the 'silent all day' problem)."""

    def setUp(self):
        self._orig_stream = pairs_candidate_campaign.STREAM_OUTPUT

    def tearDown(self):
        pairs_candidate_campaign.STREAM_OUTPUT = self._orig_stream

    def _run(self, cmd, log):
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            pairs_candidate_campaign._run_command(cmd, log, dry_run=False, label="t")
        finally:
            sys.stdout = orig
        return buf.getvalue()

    def test_stream_tees_to_console_and_log(self):
        pairs_candidate_campaign.STREAM_OUTPUT = True
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "x.log"
            out = self._run([sys.executable, "-c", "print('hello-stream')"], log)
            self.assertIn("hello-stream", out)               # streamed to console
            self.assertIn("hello-stream", log.read_text())   # also teed to log
            self.assertIn("done in", out)                    # timing line printed

    def test_capture_mode_writes_log(self):
        pairs_candidate_campaign.STREAM_OUTPUT = False
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "x.log"
            self._run([sys.executable, "-c", "print('quiet-capture')"], log)
            self.assertIn("quiet-capture", log.read_text())

    def test_nonzero_exit_raises(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "x.log"
            with self.assertRaises(RuntimeError):
                self._run([sys.executable, "-c", "import sys; sys.exit(3)"], log)

    def test_dry_run_writes_marker_only(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "x.log"
            pairs_candidate_campaign._run_command([sys.executable, "-c", "print(1)"], log, dry_run=True)
            self.assertIn("DRY RUN", log.read_text())


if __name__ == "__main__":
    unittest.main()
