"""Unit tests for tools/research_all.py pure helpers + dry-run orchestration."""
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import research_all as ra  # noqa: E402

UNIVERSES = {
    "deployed": [{"ticker": "JPM", "sector": "Financials"}, {"ticker": "BAC", "sector": "Financials"}],
    "large_cap_financials": [
        {"ticker": "JPM", "sector": "Financials"},
        {"ticker": "BAC", "sector": "Financials"},
        {"ticker": "WFC", "sector": "Financials"},
    ],
    "semiconductors": [
        {"ticker": "NVDA", "sector": "Semiconductors"},
        {"ticker": "AMD", "sector": "Semiconductors"},
    ],
    "index_etfs": [{"ticker": "SPY", "sector": "Index"}],
}


class TestSelectUniverses(unittest.TestCase):
    def test_all_minus_default_exclude(self):
        names = ra.select_universes(UNIVERSES, ["all"], list(ra._DEFAULT_EXCLUDE))
        self.assertIn("large_cap_financials", names)
        self.assertIn("semiconductors", names)
        self.assertNotIn("deployed", names)
        self.assertNotIn("index_etfs", names)

    def test_explicit_selection(self):
        names = ra.select_universes(UNIVERSES, ["semiconductors"], [])
        self.assertEqual(names, ["semiconductors"])

    def test_unknown_names_dropped(self):
        names = ra.select_universes(UNIVERSES, ["semiconductors", "does_not_exist"], [])
        self.assertEqual(names, ["semiconductors"])


class TestTickerUnion(unittest.TestCase):
    def test_union_dedupes_and_sorts(self):
        tickers = ra.ticker_union(UNIVERSES, ["deployed", "large_cap_financials"])
        self.assertEqual(tickers, ["BAC", "JPM", "WFC"])  # JPM/BAC deduped

    def test_empty(self):
        self.assertEqual(ra.ticker_union(UNIVERSES, []), [])


class TestAggregateSurvivors(unittest.TestCase):
    def _write_b3(self, path, rows):
        header = "y,x,n_windows,n_returns,total_pnl,total_trades,mean_pct,ci_lo,ci_hi,raw_p,error\n"
        path.write_text(header + "".join(rows), encoding="utf-8")

    def test_aggregate_ranks_and_flags(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fin = d / "b3_fin.csv"
            semi = d / "b3_semi.csv"
            self._write_b3(fin, [
                "JPM,BAC,47,47,2000,120,0.44,0.21,0.80,0.001,\n",   # strong candidate
                "WFC,C,47,47,-50,30,-0.01,-0.05,0.03,0.40,\n",       # not a candidate
            ])
            self._write_b3(semi, [
                "NVDA,AMD,47,47,300,40,0.10,0.02,0.20,0.030,\n",     # candidate
            ])
            rows = ra.aggregate_survivors({"large_cap_financials": fin, "semiconductors": semi}, alpha=0.05)
            self.assertEqual(len(rows), 3)
            # Sorted by raw_p asc -> JPM/BAC (0.001) first.
            self.assertEqual((rows[0]["y"], rows[0]["x"]), ("JPM", "BAC"))
            self.assertEqual(rows[0]["universe"], "large_cap_financials")
            self.assertEqual(rows[0]["candidate"], "1")
            cand = {(r["y"], r["x"]): r["candidate"] for r in rows}
            self.assertEqual(cand[("JPM", "BAC")], "1")
            self.assertEqual(cand[("NVDA", "AMD")], "1")
            self.assertEqual(cand[("WFC", "C")], "0")

    def test_missing_file_skipped(self):
        rows = ra.aggregate_survivors({"x": Path("/nonexistent/b3.csv")})
        self.assertEqual(rows, [])

    def test_dedup_same_pair_within_universe_keeps_best_p(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            b3 = d / "b3.csv"
            # DUK/EXC appears twice in one universe (the stale-append bug) with
            # different p; aggregation must keep one row (smallest raw_p).
            self._write_b3(b3, [
                "DUK,EXC,46,46,500,40,0.05,0.01,0.10,0.0028,\n",
                "DUK,EXC,46,46,500,40,0.05,0.01,0.10,0.0004,\n",
            ])
            rows = ra.aggregate_survivors({"large_cap_utilities": b3}, alpha=0.05)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["raw_p"], "0.000400")

    def test_same_pair_across_universes_kept(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            tech = d / "tech.csv"
            semi = d / "semi.csv"
            self._write_b3(tech, ["AVGO,NVDA,46,46,300,30,0.1,0.02,0.2,0.14,\n"])
            self._write_b3(semi, ["AVGO,NVDA,46,46,300,30,0.1,0.02,0.2,0.14,\n"])
            rows = ra.aggregate_survivors({"large_cap_tech": tech, "semiconductors": semi})
            # Same pair in two different universes is legitimate -> both kept.
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["universe"] for r in rows}, {"large_cap_tech", "semiconductors"})


class TestDryRun(unittest.TestCase):
    def test_dry_run_runs_no_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            yml = d / "uni.yaml"
            yml.write_text(yaml.safe_dump(UNIVERSES), encoding="utf-8")
            rc = ra.main([
                "--universes-yaml", str(yml),
                "--universes", "semiconductors",
                "--out-dir", str(d / "out"),
                "--data-dir", str(d / "data"),
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            # Dry-run writes command logs but no survivors.csv.
            self.assertTrue((d / "out" / "fetch.log").exists())
            self.assertFalse((d / "out" / "survivors.csv").exists())


if __name__ == "__main__":
    unittest.main()
