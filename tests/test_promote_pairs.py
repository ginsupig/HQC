"""Unit tests for tools/promote_pairs.py — the selection->deploy bridge.

These run on synthetic campaign artifacts (no market data, no statsmodels),
exercising the strict gate, parameter resolution, YAML rendering, and the
dry-run / write / refuse-empty behaviour of main().
"""
import sys
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import promote_pairs as pp  # noqa: E402


def _row(pair, bucket="APPROVED", bonf=True, ci_lo=0.05, ci_hi=0.5,
         raw_p=0.001, gates=("PASS", "PASS", "PASS", "PASS"), mean_pct=0.2):
    return pp.CandidateRow(
        pair=pair,
        bucket=bucket,
        mean_pct=mean_pct,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        raw_p=raw_p,
        bonferroni_pass=bonf,
        bonferroni_threshold="0.00714",
        gates=dict(zip(pp._REQUIRED_GATES, gates)),
    )


CAMPAIGN_YAML = """
campaign:
  output_dir: state/test_campaign
  pre_registered_family_size: 5
  pair_entry_z: 1.5
  pair_exit_z: 0.4
  pair_target_dollar_notional: 10000.0
pairs:
  - y: JPM
    x: BAC
    thesis: "Bank duopoly."
  - y: V
    x: MA
    entry_z: 2.0
    target_dollar_notional: 5000.0
    thesis: "Payment duopoly."
  - y: KO
    x: PEP
    thesis: "Staples."
"""

RANKING_CSV = (
    "pair,bucket,mean_pct,ci_lo,ci_hi,raw_p,bonferroni_pass,bonferroni_threshold,a2,a3,a4,d3,artifacts_dir\n"
    "JPM/BAC,APPROVED,0.44,0.21,0.82,0.001,True,0.00714,PASS,PASS,PASS,PASS,x\n"
    "V/MA,PROBATION,0.10,0.01,0.20,0.03,True,0.00714,PASS,ATTENTION,PASS,PASS,x\n"
    "KO/PEP,REJECTED,-0.01,-0.05,0.03,0.4,False,0.00714,FAIL,PASS,PASS,PASS,x\n"
)


class TestStrictGate(unittest.TestCase):
    def test_clean_approved_passes(self):
        self.assertEqual(_row("A/B").strict_failures(), [])

    def test_bucket_must_be_approved(self):
        self.assertIn("bucket=PROBATION", _row("A/B", bucket="PROBATION").strict_failures())

    def test_bonferroni_required(self):
        self.assertIn("bonferroni_pass=False", _row("A/B", bonf=False).strict_failures())

    def test_ci_lo_must_exceed_zero(self):
        self.assertTrue(any("ci_lo" in r for r in _row("A/B", ci_lo=0.0).strict_failures()))
        self.assertTrue(any("ci_lo" in r for r in _row("A/B", ci_lo=-0.1).strict_failures()))

    def test_every_gate_must_pass(self):
        fails = _row("A/B", gates=("PASS", "ATTENTION", "PASS", "PASS")).strict_failures()
        self.assertIn("a3=ATTENTION", fails)


class TestSelection(unittest.TestCase):
    def _campaign(self, tmp):
        path = tmp / "campaign.yaml"
        path.write_text(CAMPAIGN_YAML, encoding="utf-8")
        return pp.load_campaign_params(path)

    def test_load_campaign_params_resolves_overrides(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            defaults, per_pair, output_dir = self._campaign(Path(d))
            self.assertEqual(defaults["entry_z"], 1.5)
            self.assertEqual(output_dir.name, "test_campaign")
            # V/MA overrides entry_z and notional
            vma = pp.resolve_params("V/MA", defaults, per_pair)
            self.assertEqual(vma["entry_z"], 2.0)
            self.assertEqual(vma["target_dollar_notional"], 5000.0)
            self.assertEqual(vma["exit_z"], 0.4)  # falls back to campaign default
            # JPM/BAC uses all defaults
            jpm = pp.resolve_params("JPM/BAC", defaults, per_pair)
            self.assertEqual(jpm["entry_z"], 1.5)
            self.assertEqual(jpm["target_dollar_notional"], 10000.0)

    def test_select_only_approved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            defaults, per_pair, _ = self._campaign(Path(d))
            ranking = [
                _row("JPM/BAC"),
                _row("V/MA", bucket="PROBATION", gates=("PASS", "ATTENTION", "PASS", "PASS")),
                _row("KO/PEP", bucket="REJECTED", bonf=False, ci_lo=-0.05,
                     gates=("FAIL", "PASS", "PASS", "PASS")),
            ]
            approved, warnings = pp.select_approved(ranking, defaults, per_pair)
            self.assertEqual([dp.label for dp in approved], ["JPM/BAC"])
            self.assertEqual(warnings, [])

    def test_mislabeled_approved_is_refused_with_warning(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            defaults, per_pair, _ = self._campaign(Path(d))
            # Bucket says APPROVED but a gate is ATTENTION -> must refuse + warn.
            bad = _row("JPM/BAC", gates=("PASS", "ATTENTION", "PASS", "PASS"))
            approved, warnings = pp.select_approved([bad], defaults, per_pair)
            self.assertEqual(approved, [])
            self.assertTrue(any("REFUSED JPM/BAC" in w for w in warnings))

    def test_approved_but_absent_from_campaign_is_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            defaults, per_pair, _ = self._campaign(Path(d))
            approved, warnings = pp.select_approved([_row("XOM/CVX")], defaults, per_pair)
            self.assertEqual(approved, [])
            self.assertTrue(any("absent from campaign" in w for w in warnings))


class TestRender(unittest.TestCase):
    def _approved(self):
        return [
            pp.DeployPair(
                y="JPM", x="BAC",
                params=dict(pp._DEFAULT_PARAMS),
                thesis="Bank duopoly.",
                provenance={"raw_p": "0.001000", "ci_lo": "0.21", "ci_hi": "0.82",
                            "mean_pct": "0.44", "bonferroni_threshold": "0.00714",
                            "gates": "a2=PASS a3=PASS a4=PASS d3=PASS"},
            )
        ]

    def test_render_roundtrips_and_preserves_meta(self):
        existing = {"paper": False, "feed": "sip",
                    "risk": {"initial_capital": 250000.0, "daily_loss_pct_kill": 0.03,
                             "beta_drift_pct_kill": 0.25, "beta_drift_window_min": 90}}
        text = pp.render_yaml(self._approved(), existing, "config/pairs_research.yaml",
                              "state/test_campaign/ranking.csv")
        loaded = yaml.safe_load(text)
        self.assertEqual(loaded["paper"], False)  # preserved, never flipped
        self.assertEqual(loaded["feed"], "sip")
        self.assertEqual(loaded["risk"]["initial_capital"], 250000.0)
        self.assertEqual(loaded["risk"]["beta_drift_window_min"], 90)
        self.assertEqual(len(loaded["pairs"]), 1)
        self.assertEqual(loaded["pairs"][0]["y"], "JPM")
        self.assertEqual(loaded["pairs"][0]["entry_z"], 1.5)
        # provenance comment present
        self.assertIn("raw_p=0.001000", text)
        self.assertIn("thesis: Bank duopoly.", text)

    def test_render_emits_nondefault_delta_ve(self):
        params = dict(pp._DEFAULT_PARAMS)
        params["delta"] = 5e-5
        pair = pp.DeployPair(y="V", x="MA", params=params, thesis="",
                             provenance={k: "" for k in
                                         ("raw_p", "ci_lo", "ci_hi", "mean_pct",
                                          "bonferroni_threshold", "gates")})
        text = pp.render_yaml([pair], {}, "c", "r")
        loaded = yaml.safe_load(text)
        self.assertAlmostEqual(loaded["pairs"][0]["delta"], 5e-5)
        self.assertNotIn("ve:", text)  # ve at default -> omitted

    def test_gross_exposure_warning(self):
        pairs = [
            pp.DeployPair(y="A", x="B", params={**pp._DEFAULT_PARAMS, "target_dollar_notional": 40000.0},
                          thesis="", provenance={}),
            pp.DeployPair(y="C", x="D", params={**pp._DEFAULT_PARAMS, "target_dollar_notional": 40000.0},
                          thesis="", provenance={}),
        ]
        # gross = 2 * (40k + 40k) = 160k. At 1x of 100k buying power -> warn.
        gross, limit, warning = pp.gross_exposure_report(pairs, buying_power=100000.0, max_leverage=1.0)
        self.assertEqual(gross, 160000.0)
        self.assertEqual(limit, 100000.0)
        self.assertIsNotNone(warning)

    def test_gross_exposure_ok_under_leverage(self):
        pairs = [
            pp.DeployPair(y="A", x="B", params={**pp._DEFAULT_PARAMS, "target_dollar_notional": 40000.0},
                          thesis="", provenance={}),
            pp.DeployPair(y="C", x="D", params={**pp._DEFAULT_PARAMS, "target_dollar_notional": 40000.0},
                          thesis="", provenance={}),
        ]
        # 160k gross is well under 450k buying power at 3x (1.35M) -> no warning.
        gross, limit, warning = pp.gross_exposure_report(pairs, buying_power=450000.0, max_leverage=3.0)
        self.assertEqual(limit, 1350000.0)
        self.assertIsNone(warning)


class TestMainEndToEnd(unittest.TestCase):
    def _setup(self, tmp: Path):
        camp = tmp / "campaign.yaml"
        camp.write_text(CAMPAIGN_YAML, encoding="utf-8")
        out_dir = _REPO_ROOT / "state" / "test_campaign"
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking = out_dir / "ranking.csv"
        ranking.write_text(RANKING_CSV, encoding="utf-8")
        target = tmp / "pairs.yaml"
        return camp, ranking, target

    def test_dry_run_writes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            camp, _, target = self._setup(Path(d))
            rc = pp.main(["--campaign-config", str(camp), "--target", str(target)])
            self.assertEqual(rc, 0)
            self.assertFalse(target.exists())

    def test_write_promotes_only_approved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            camp, _, target = self._setup(Path(d))
            rc = pp.main(["--campaign-config", str(camp), "--target", str(target), "--write"])
            self.assertEqual(rc, 0)
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            labels = [f"{p['y']}/{p['x']}" for p in loaded["pairs"]]
            self.assertEqual(labels, ["JPM/BAC"])  # V/MA probation, KO/PEP rejected
            self.assertEqual(loaded["paper"], True)  # default when no existing target

    def test_keep_existing_unions_current_basket(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            camp, _, target = self._setup(Path(d))
            target.write_text(yaml.safe_dump({
                "paper": True, "feed": "iex",
                "pairs": [{"y": "GOOG", "x": "GOOGL", "entry_z": 1.5, "exit_z": 0.4,
                           "cooldown_seconds": 120, "nominal_stop_pct": 0.02,
                           "target_dollar_notional": 10000}],
                "risk": {"initial_capital": 100000},
            }), encoding="utf-8")
            rc = pp.main(["--campaign-config", str(camp), "--target", str(target),
                          "--write", "--keep-existing"])
            self.assertEqual(rc, 0)
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            labels = sorted(f"{p['y']}/{p['x']}" for p in loaded["pairs"])
            self.assertEqual(labels, ["GOOG/GOOGL", "JPM/BAC"])

    def test_refuse_empty_basket(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            camp = Path(d) / "campaign.yaml"
            camp.write_text(CAMPAIGN_YAML, encoding="utf-8")
            out_dir = _REPO_ROOT / "state" / "test_campaign"
            out_dir.mkdir(parents=True, exist_ok=True)
            # All rows non-approvable.
            (out_dir / "ranking.csv").write_text(
                "pair,bucket,mean_pct,ci_lo,ci_hi,raw_p,bonferroni_pass,bonferroni_threshold,a2,a3,a4,d3,artifacts_dir\n"
                "KO/PEP,REJECTED,-0.01,-0.05,0.03,0.4,False,0.00714,FAIL,PASS,PASS,PASS,x\n",
                encoding="utf-8")
            target = Path(d) / "pairs.yaml"
            rc = pp.main(["--campaign-config", str(camp), "--target", str(target)])
            self.assertEqual(rc, 1)  # refused
            self.assertFalse(target.exists())

    def tearDown(self):
        import shutil
        shutil.rmtree(_REPO_ROOT / "state" / "test_campaign", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
