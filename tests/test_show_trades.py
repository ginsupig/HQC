import json
import tempfile
import unittest
from pathlib import Path

import tools.show_trades as st


class TestShowTrades(unittest.TestCase):
    def _write(self, path: Path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_load_jsonl_skips_blank_and_corrupt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "outcomes.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": "2026-06-08T14:00:00+00:00"}) + "\n")
                f.write("\n")
                f.write("{not json}\n")
                f.write(json.dumps({"ts": "2026-06-08T15:00:00+00:00"}) + "\n")
            self.assertEqual(len(st.load_jsonl(p)), 2)

    def test_load_missing_file_returns_empty(self):
        self.assertEqual(st.load_jsonl(Path("/nope/outcomes.jsonl")), [])

    def test_filter_by_date(self):
        rows = [
            {"ts": "2026-06-08T14:00:00+00:00", "symbol": "PNC"},
            {"ts": "2026-06-07T20:00:00+00:00", "symbol": "USB"},
            {"ts": "2026-06-08T19:30:00+00:00", "symbol": "PNC"},
        ]
        self.assertEqual(len(st.filter_by_date(rows, "2026-06-08")), 2)
        self.assertEqual(len(st.filter_by_date(rows, None)), 3)

    def test_summarize_counts_only_closed_and_wins(self):
        outcomes = [
            {"net_pnl": 12.5, "gross_pnl": 14.0},   # win, closed
            {"net_pnl": -4.0, "gross_pnl": -3.0},   # loss, closed
            {"net_pnl": None},                       # open leg, not closed
            {"status": "filled"},                    # fill with no pnl
        ]
        s = st.summarize(outcomes)
        self.assertEqual(s["fills"], 4)
        self.assertEqual(s["closed"], 2)
        self.assertEqual(s["wins"], 1)
        self.assertAlmostEqual(s["net_pnl"], 8.5)
        self.assertAlmostEqual(s["gross_pnl"], 11.0)
        self.assertAlmostEqual(s["win_rate"], 0.5)

    def test_summarize_empty(self):
        s = st.summarize([])
        self.assertEqual(s["closed"], 0)
        self.assertEqual(s["win_rate"], 0.0)

    def test_main_runs_on_empty_day(self):
        # No file present -> still returns 0 and prints the no-trades hint.
        with tempfile.TemporaryDirectory() as d:
            rc = st.main(["--root", d, "--date", "2026-06-08"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
