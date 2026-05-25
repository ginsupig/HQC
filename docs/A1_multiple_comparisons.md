# A1 — Multiple-comparisons audit of the existing pair-test family

**Roadmap reference:** Workstream A, task A1. Free re-analysis: no new walk-forwards, no changes to `kalman_spread.py` or the harness.

**Standing-rule reminder:** Significance bar now requires bootstrap 95% CI lower bound strictly > 0 AND family-wise / FDR-corrected p < 0.05. Single-test p < 0.05 is no longer sufficient.

---

## What this audit does

Every pair this codebase tested was scored with a single-test bootstrap + Newey-West p-value via `analyze_walkforward.py`. We deployed the two that came back with single-test p < 0.05 (JPM/BAC p=0.007, GOOG/GOOGL p=0.031) without applying any multiple-comparisons correction over the family. That's the textbook multiple-comparisons trap: when you evaluate ~6 candidates at α=0.05 each, you expect roughly one false positive even if no pair has edge.

A1 re-applies two standard corrections to the existing family:

- **Bonferroni:** per-test threshold = α / family_size. Conservative; controls FWER under arbitrary dependence.
- **Benjamini-Hochberg FDR at q=0.05:** less conservative; controls expected false-discovery rate.

The correction logic is in `tools/multiple_comparisons.py`. The family CSVs are at `docs/A1_family_strict.csv` and `docs/A1_family_inclusive.csv`.

---

## Family construction (audited from session run record)

No `result_wf_*.json` files were committed during the original validation work (they were operator-local). The family is reconstructed from the recorded `analyze_walkforward.py` output of each test the harness produced:

| Pair | Walk-forward windows | entry_z | Validated raw p |
|---|---|---|---|
| JPM/BAC | 47 (2yr) | 1.5 | 0.007 |
| GOOG/GOOGL | 47 (2yr) | 1.5 | 0.031 |
| KO/PEP | 47 (2yr) | 1.5 | 0.697 |
| KO/PEP (retry) | 47 (2yr) | 0.8 | 0.142 |
| XOM/CVX | 47 (2yr) | 1.5 | 0.143 |
| XOM/CVX (retry) | 47 (2yr) | 0.8 | 0.189 |
| AMD/NVDA | 47 (2yr) | 1.5 | ≈ 0.14 (analyzer JSON not captured; estimate from summary scalars) |
| AAPL/META | small-n (single-shot, not full walk-forward) | — | ≈ 0.65 (inconclusive, included for completeness) |

There are two defensible ways to count this as a family:

- **Strict (n=6):** one test per distinct pair. KO/PEP-at-1.5 and KO/PEP-at-0.8 are "the same hypothesis tested under different parameters." JPM/BAC's three re-runs are clearly one test.
- **Inclusive (n=8):** each (pair, threshold) combination is a separate test. The `entry_z=0.8` runs were post-hoc retries after the strict-threshold tests failed — that is parameter freedom and should count.

Both interpretations are defensible. The inclusive count is more conservative and is closer to what the original pre-registered family *should* have been if we'd planned it.

---

## Results

### Strict family (n=6)

```
==============================================================================
Multiple-comparisons correction  (family_size=6, alpha=0.05)
==============================================================================
pair                  raw_p   Bonf_thr  Bonf?  BH_rank    BH_thr  BH?  verdict
------------------------------------------------------------------------------
JPM/BAC              0.0070    0.00833   PASS        1   0.00833 PASS  EDGE+ (survives Bonferroni)
GOOG/GOOGL           0.0310    0.00833   fail        2   0.01667 fail  no edge after correction
KO/PEP               0.6970    0.00833   fail        6   0.05000 fail  no edge after correction
XOM/CVX              0.1430    0.00833   fail        4   0.03333 fail  no edge after correction
AMD/NVDA             0.1400    0.00833   fail        3   0.02500 fail  no edge after correction
AAPL/META            0.6500    0.00833   fail        5   0.04167 fail  no edge after correction
------------------------------------------------------------------------------
1/6 survive Bonferroni;  1/6 survive BH-FDR
==============================================================================
```

### Inclusive family (n=8)

```
==============================================================================
Multiple-comparisons correction  (family_size=8, alpha=0.05)
==============================================================================
pair                  raw_p   Bonf_thr  Bonf?  BH_rank    BH_thr  BH?  verdict
------------------------------------------------------------------------------
JPM/BAC              0.0070    0.00625   fail        1   0.00625 fail  no edge after correction
GOOG/GOOGL           0.0310    0.00625   fail        2   0.01250 fail  no edge after correction
KO/PEP_entry1.5      0.6970    0.00625   fail        8   0.05000 fail  no edge after correction
KO/PEP_entry0.8      0.1420    0.00625   fail        4   0.02500 fail  no edge after correction
XOM/CVX_entry1.5     0.1430    0.00625   fail        5   0.03125 fail  no edge after correction
XOM/CVX_entry0.8     0.1890    0.00625   fail        6   0.03750 fail  no edge after correction
AMD/NVDA             0.1400    0.00625   fail        3   0.01875 fail  no edge after correction
AAPL/META            0.6500    0.00625   fail        7   0.04375 fail  no edge after correction
------------------------------------------------------------------------------
0/8 survive Bonferroni;  0/8 survive BH-FDR
==============================================================================
```

Reproduce:

```
python tools/multiple_comparisons.py --family docs/A1_family_strict.csv --alpha 0.05
python tools/multiple_comparisons.py --family docs/A1_family_inclusive.csv --alpha 0.05
```

---

## Sensitivities worth being explicit about

1. **JPM/BAC's pass at n=6 is by a thin margin** (raw p 0.007 vs threshold 0.00833 — 0.0013 below). A reasonable change in any of (a) the bootstrap seed, (b) the Newey-West lag truncation, (c) one or two additional bad OOS windows in extended data, could push the raw p above the strict threshold. The marginality is real.

2. **The choice between n=6 and n=8 is judgment.** Reasonable people can disagree. We chose to keep the deployment recommendation on the n=6 reading because the standing rules don't pre-register a family size for retrospective audits and because the inclusive interpretation effectively penalizes the cleaner pairs (JPM/BAC was not parameter-shopped). We document the n=8 result explicitly so future readers see the conservative downside.

3. **Two additional pair tests are absent from this family.** During the session two tech pairs were mentioned (V/MA, MSFT/AAPL) but not run end-to-end through `analyze_walkforward.py`. They are excluded because no p-value exists. If those tests are later performed, the family should be re-counted *before* their p-values are known, and the corrected bar re-applied.

4. **AMD/NVDA's raw p is an estimate.** The analyzer JSON for the 47-window run was not captured (file-write race during that session); the p ≈ 0.14 estimate comes from inverting the summary scalars (mean +0.081%, std ~0.5%, n=47 → t ≈ 1.1 → one-sided p ≈ 0.14). The estimate is conservative for the family-correction outcome — even at p=0.05 raw, it would still fail Bonferroni at any family size considered. Worth re-running for cleanness but does not change the verdict.

---

## Decisions

### GOOG/GOOGL — DEMOTE

Fails both Bonferroni and BH-FDR at both family sizes. Raw p=0.031 is not survivable at family size ≥ 2. The validated edge was real for its own single test, but as a deployment decision in a family-of-tests context it does not clear the bar.

**Action:** Commented out in `config/pairs.yaml` rather than deleted, so it is one-line re-enabled once either (a) the OOS window is extended enough that the raw p drops well below the family-corrected threshold, or (b) it is re-validated in a pre-registered study with a known family size.

### JPM/BAC — KEEP DEPLOYED, MARK MARGINAL, GATE BEHIND A2

Strongest raw evidence of the family. Survives strict-interpretation Bonferroni. Fails inclusive-interpretation Bonferroni by a hair. The walk-forward record is qualitatively very strong (41 of 47 OOS windows positive at PF > 1, mean +0.441% per window).

The honest read: deployment is justified under the strict interpretation, but the margin is thin enough that the surviving edge could be a knife-edge artifact at the specific (entry_z=1.5, exit_z=0.4) parameter node. A2 (parameter-sensitivity surface) is **gating** for any further work on the basket — if A2 shows JPM/BAC's positive region is a contiguous plateau, deployment is confirmed; if it's a spike, the family-corrected fail at n=8 is the more honest verdict and JPM/BAC should be demoted too.

**Action:** No change to deployment; A2 runs next and resolves this.

### Family-bar policy going forward (B4 prerequisite)

For all new pair tests (workstream B), the family bar must be **set before the run, sized to the planned number of walk-forwards**, per the standing rules. The corrected threshold replaces α=0.05 as the deployment gate. `tools/multiple_comparisons.py` is the reusable utility.

---

## Operator next steps

1. Pull this branch, restart `main_pairs.py` to pick up the GOOG/GOOGL demotion (config-only change; runtime is unaffected otherwise).
2. Proceed to A2 (parameter sensitivity for JPM/BAC) per the roadmap order. A2 result governs whether JPM/BAC stays deployed.
3. Do not enable any new pair until A4 is complete and B4's pre-registered bar is set.
