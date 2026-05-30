import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
