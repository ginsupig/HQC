# B4 — Pre-registered corrected significance bar for the B3 pair family

**Status:** DRAFT (rule frozen; K to be filled in once B2 output is known
and BEFORE B3 per-pair results are inspected).

**Roadmap reference:** Workstream B, task B4. This document **commits the
significance rule** for Workstream B's pair-discovery family in advance of
seeing B3 results, so the corrected-p-value threshold cannot be tuned to
match whichever pairs happened to come out of B3 looking pretty.

Editing the rule section of this file after the first B3 per-pair p-value
is inspected is a protocol violation. The K section may be filled in once
between B2 and B3; the rule section is locked at commit time.

Commit hash at K-freeze: ________________ (fill on K-freeze commit)
Date of K-freeze: ________________
Author: Adrian

---

## 0. What this document locks

The corrected significance bar applied to per-pair p-values produced by
`tools/b3_batch_walkforward.py`. Without this commitment, B3's per-pair
`raw_p` column would tempt post-hoc "the surviving pairs are clearly real"
narratives that don't survive a multiple-testing correction.

Two things are locked **here** (in this commit):

1. The **rule**: family-wise correction method, alpha, CI requirement,
   pooled-vs-per-pair interpretation.
2. The **inputs the rule consumes**: which p-value column from B3, how
   K is counted, what "pass" means.

One thing is locked **separately**, in the K-freeze commit:

3. The **value of K**: number of pairs B3 will score. K must be filled
   in after B2 runs and before any B3 per-pair p-value is inspected.

---

## 1. The rule (locked now)

### 1.1 Significance bar (primary: Bonferroni)

A per-pair B3 result is **EDGE+** only when ALL of:

- `b3_raw_p < alpha / K`  (Bonferroni-corrected p-value)
- `b3_ci_lo > 0`           (bootstrap 95% CI lower bound strictly above zero)
- `b3_error == ""`         (no worker error)

Where:
- `alpha = 0.05`           (frozen here; tighter alpha is permitted as a
                            secondary check, but the 0.05 bar is the
                            decision bar)
- `K = <family size>`      (locked in §2 below at K-freeze time)
- `b3_*` columns refer to the per-pair output of
  `tools/b3_batch_walkforward.py`.

### 1.2 Secondary bar (BH-FDR, reported only)

Benjamini–Hochberg FDR at `q = 0.05` is computed and reported alongside
Bonferroni, but is **not** the deployment gate. Reasoning: directional
pair-trading edges should clear FWER, not just FDR, to justify live
capital. BH is informational.

### 1.3 Pooled verdict (the family-level check)

In addition to per-pair EDGE+, the family must clear a pooled bar:

- Pool the per-pair `mean_pct` values (one observation per pair).
- Compute `pooled_p = Newey-West HAC one-sample p(mean > 0)` on the
  pair-mean distribution (this is what `b3_batch_walkforward.py` prints
  in its summary block).
- The family is **POOLED EDGE+** only when:
  - `pooled_p < alpha`  (uncorrected; the pooled mean is one test)
  - `pooled_ci_lo > 0`  (bootstrap 95% CI lower bound on the pooled mean)

This is the F1 §3.1 / roadmap rule applied here: deployment decisions use
pooled verdicts as the family-level unit, and per-pair Bonferroni gates
which specific pairs are eligible *within* a POOLED EDGE+ family.

### 1.4 Combined deployment rule

A pair from the B3 family is eligible for paper-trading prep only if **all
three** conditions hold:

1. The family is **POOLED EDGE+** (§1.3).
2. The individual pair is **EDGE+** (§1.1 — Bonferroni at K).
3. A3-style cost stress at `slippage_bps_per_side = 3.0` has been run on
   the pair and it survives the per-pair EDGE+ bar at the stressed cost.
   (A3 already demonstrated JPM/BAC is slippage-fragile; same gate applies
   to every new pair that survives this funnel.)

If the family is **NOT POOLED EDGE+**, no per-pair result deploys regardless
of how attractive individual `raw_p` values look — that would be cherry-picking.

---

## 2. K and per-pair threshold (locked at K-freeze commit)

K is the number of pairs B3 scored. Filled in by appending to this file
in a separate commit between B2 and B3.

| field | value |
|---|---|
| K (pairs scored by B3) | **TBD** |
| Universe(s) included | TBD (e.g., `large_cap_financials`) |
| B2 thresholds used | `coint_p_max = 0.05`, `max_half_life_days = 60` |
| B2 output file (sha256) | TBD |
| Whether deployed pair (JPM/BAC) is in the family | TBD (yes/no — affects K by 1) |
| Bonferroni per-pair threshold = 0.05 / K | TBD |
| Bonferroni at alpha=0.01 (tighter check) | TBD |

K-freeze checklist (do once, in order, no skipping):

- [ ] B1 ran on the chosen universe(s); output file recorded.
- [ ] B2 ran on B1 output; `screen_pass=1` row count counted.
- [ ] K decided (with or without JPM/BAC). Decision documented.
- [ ] This file updated with K, dates, sha256s.
- [ ] Commit message includes the K-freeze label and references this file.
- [ ] **Only then** is `tools/b3_batch_walkforward.py` run.

After K-freeze:

- [ ] B3 runs.
- [ ] Per-pair `raw_p` compared against `0.05 / K`.
- [ ] Pooled summary checked against §1.3.
- [ ] Eligible pairs (per §1.4) flagged for paper-trade prep.

---

## 3. What this document does NOT do

- It does **not** decide which universes to run B1 on. That is a separate
  pre-registration if a new universe is added — see B1 docs.
- It does **not** approve cross-family pooling. The leveraged sub-family
  in F1 §1.2 is a separate family with its own K; B3's K is the
  unleveraged-pairs family.
- It does **not** loosen for marginal misses. A pair at `raw_p = 0.011`
  when `0.05/K = 0.010` is **not** EDGE+; it is a close miss and stays in
  the inconclusive bucket.

---

## 4. Anti-gaming rules

1. **The rule is locked at this commit's hash.** Changes to §1 require a
   new pre-registration (B4b) with its own commit and family-size
   accounting; you cannot edit B4 to make a result pass after the fact.
2. **K is locked at K-freeze commit, not at B3-result-inspection.** Once
   any B3 per-pair p-value is read, K is final.
3. **No "best K" selection.** K is the count of pairs B3 actually scored
   under the locked B2 thresholds. You cannot retroactively narrow the
   universe to shrink K and make Bonferroni easier.
4. **No cost-axis double-dipping.** §1.4 requires the A3-stress survival
   check at 3 bps slippage on each individual surviving pair. Skipping
   this check because the un-stressed result is pretty is a violation.
5. **Disconfirmation is acceptable.** If the family is not POOLED EDGE+,
   that is a valid published result, not a reason to retry.

---

## 5. Why Bonferroni and not BH-FDR for the deployment gate

Bonferroni controls family-wise error rate (FWER) under arbitrary
dependence — appropriate for pair-trading candidates, which are
correlated through shared market factors. BH-FDR controls the expected
false-discovery proportion among rejections; appropriate for high-throughput
hypothesis testing where some false positives are tolerable (e.g.,
genome-wide screens). For live capital deployment, the cost of one false
positive (a deployed pair with no edge eating losses) is high enough that
FWER is the right framing. BH is reported as a sanity check.

This is consistent with A1's choice for the existing-edge audit
(`tools/multiple_comparisons.py` reports both; the deployment narrative
uses Bonferroni).
