# B3 — Batch walk-forward across B2 survivors

**Roadmap reference:** Workstream B, task B3. Runs the deployed Kalman pairs
strategy through the existing walk-forward harness on every pair B2 flagged
`screen_pass = 1`, captures per-window OOS returns, and aggregates a pooled
verdict. This is the **last research step before A1's family-corrected
deployment bar**.

---

## Pipeline position

```
B1 (correlation)        -> b1_*.csv
B2 (cointegration)      -> b2_*.csv  (screen_pass=1 rows)
B3 (THIS TOOL)          -> b3_*.csv  + b3_*.returns.json
B4 (pre-register bar)   -> docs/B4_*.md  (family-size = K locked BEFORE inspecting B3 p-values)
A1 (corrected verdict)  -> deploy / demote
```

## What B3 does

1. Reads B2's output CSV. By default scores only `screen_pass = 1` rows;
   `--all-rows` scores everything (useful to see what *almost* passed).
2. For each surviving pair, locates the y and x leg CSVs in `--data-dir`.
3. Runs walk-forward at a **fixed parameter set** (default = deployed config:
   entry_z=1.5, exit_z=0.4, δ=1e-4, Ve=1e-3, 30/10 train/test, 1.5 bps
   slippage, 0.25% borrow APR, 240-min hard hold).
4. Per pair: bootstrap 95% CI on mean per-window OOS return; Newey-West HAC
   one-sample p(mean>0).
5. Pooled: per-pair means as observations → pooled mean, bootstrap CI,
   Newey-West p. This is the family-corrected unit of analysis per the
   roadmap rule that "decision is pooled across symbols at a fixed config."
6. Writes per-pair CSV + per-pair-returns JSON for B4 / A1.

## What B3 does NOT do

- **No parameter sweep.** A2 covers that on a single pair. B3 tests parameter
  *stability* across pairs at one config.
- **No deployment decision.** A1's corrected bar (Bonferroni or BH-FDR at
  family-size = K) is the gate, and B4 must lock that bar **before** B3
  per-pair p-values are inspected.
- **No re-screening.** B2's `screen_pass=1` is trusted as-is.

---

## How to run

```bash
# Standard run: score every B2 survivor at the deployed parameter set
python tools/b3_batch_walkforward.py \
    --input b2_financials.csv \
    --data-dir data/alpaca \
    --workers 4 \
    --output b3_financials.csv
```

Resume an interrupted run (same `--output`):

```bash
python tools/b3_batch_walkforward.py \
    --input b2_financials.csv \
    --workers 4 \
    --resume \
    --output b3_financials.csv
```

To stress-test under realistic costs (matches A3's flagged-fragile slippage):

```bash
python tools/b3_batch_walkforward.py \
    --input b2_financials.csv \
    --slippage-bps-per-side 3.0 \
    --output b3_financials_3bps.csv
```

## Outputs

`b3_*.csv` — one row per pair:

| column | meaning |
|---|---|
| `y`, `x` | Pair members carried from B2. |
| `n_windows` | Walk-forward windows attempted (depends on overlap). |
| `n_returns` | Windows that produced a finite OOS return. |
| `total_pnl` | Sum of realized PnL across all OOS windows ($). |
| `total_trades` | Trade count across all OOS windows. |
| `mean_pct` | Mean per-window OOS return (%). |
| `ci_lo`, `ci_hi` | Bootstrap 95% CI on mean (%). |
| `raw_p` | Newey-West HAC one-sample p(mean > 0). |
| `error` | Non-empty if the worker crashed for this pair. |

`b3_*.returns.json` — raw per-window returns per pair (for B4 + A1).

A pooled summary block is printed at the end:

```
B3 pooled verdict (family-size unit of analysis)
  K = 4 pairs scored
  pooled mean per-window return = +0.1234%
  bootstrap 95% CI on pooled mean: [+0.0123%, +0.2345%]
  Newey-West p(mean > 0) on pooled distribution = 0.00321

  Family-corrected thresholds (Bonferroni at family-size = K):
    alpha=0.05    -> per-pair threshold = 0.01250
    alpha=0.01    -> per-pair threshold = 0.00250
```

## Family-size accounting

`K` = the number of pairs B3 scored. **A1's corrected p-value bar uses
family-size = K.** If you want the deployed pair (JPM/BAC) to count toward
the family alongside B3's new candidates, K must include it.

**Pre-register K with B4 before reading B3 per-pair p-values.** Otherwise
the bar is post-hoc and the corrected p-value is gameable. The pooled
verdict in the summary block is intentionally the headline number; the
per-pair p-values are diagnostic.

## Reading the result

| Pooled `p(mean > 0)` | Pooled CI lo | Verdict |
|---|---|---|
| < Bonferroni threshold | > 0 | EDGE+ family; safe to read individual pair p-values vs the corrected bar |
| < 0.05, > Bonferroni | > 0 | Family is EDGE+ at BH-FDR but not Bonferroni — report both, prefer Bonferroni for deployment |
| ≥ 0.05 | — | EDGE− or inconclusive. Demote any new candidates; deployed pair survives on prior validation only. |
| any | ≤ 0 | EDGE− regardless of p-value; CI floor below zero is the binding constraint. |

## What comes after B3

- **B4**: pre-register the Bonferroni / BH-FDR thresholds at family-size = K
  in a committed doc, **before** acting on the per-pair p-values from B3.
- **A1 re-run** at the new family size: only pairs whose per-pair `raw_p`
  passes the corrected threshold AND whose `ci_lo > 0` move forward to
  paper-trade preparation.

## Smoke test

Sanity-check that B3 reproduces the deployed-pair walk-forward number:

```bash
# Use the deployed universe (just JPM/BAC) with default deployed params
python tools/b1_universe_clustering.py --universe deployed --output b1_d.csv
python tools/b2_candidate_screen.py --input b1_d.csv --output b2_d.csv
python tools/b3_batch_walkforward.py --input b2_d.csv --output b3_d.csv
cat b3_d.csv
```

Expected: one row, JPM/BAC, with the same per-window-return distribution
as the standalone `walkforward_pairs.py --pair-entry-z 1.5 --pair-exit-z 0.4`
output. If B3's per-pair `mean_pct` doesn't match the standalone runner,
there's a parameter wiring bug.
