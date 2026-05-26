# A3 — Cost stress / breakeven

**Roadmap reference:** Workstream A, task A3. Orthogonal to A2: A2 perturbs strategy parameters, A3 perturbs **cost assumptions**. Together they bound the robustness story for any pair we deploy.

**Standing-rule recap:** Significance bar requires bootstrap 95% CI lower bound > 0 AND family-corrected p < 0.05. The family in A3 is the cost grid (9 cells per pair) — Bonferroni per-cell threshold = `α / 9 = 0.00556` at `α=0.05`.

---

## What A3 measures

Holds the deployed strategy parameters fixed (`entry_z=1.5, exit_z=0.4, δ=1e-4, Ve=1e-3`) and varies just the cost model. For each cell in the cost grid, runs the full validated walk-forward (47 OOS windows on 2 years of pair history) and records:

- OOS mean per-window return
- Bootstrap 95% CI on the mean
- Newey-West HAC p-value
- Total realized PnL and total realized costs

Default grid (from the roadmap):

| Parameter | Values | Count |
|---|---|---|
| `slippage_bps_per_side` | 1.5, 3.0, 5.0 | 3 |
| `short_borrow_apr` | 0.0025 (0.25%), 0.01 (1%), 0.03 (3%) | 3 |
| **cells per pair** | | **9** |

Each cell = 47-window walk-forward → ~420 backtests per pair → ~10–15 min with 8 workers, ~90 min single-process.

---

## Verdict rules

A cell counts as **EDGE+** under the corrected bar when BOTH:

1. Bootstrap 95% CI lower bound > 0
2. Raw p ≤ Bonferroni threshold = `α / family_size = 0.05 / 9 = 0.00556`

The analyzer reports two things per pair:

### 1. Breakeven boundary

Walks each axis (slippage, borrow) upward from the deployed cell and finds the first cell where EDGE+ flips to fail. Reports the gap above deployed assumptions.

### 2. Fragility flag

Per the roadmap's "Flag if any deployed pair is fragile to a 1bp slippage increase":

| Axis | Fragility threshold |
|---|---|
| slippage | breakeven within `+1.0 bps` of deployed (i.e., fails at ≤ 2.5 bps/side) |
| borrow | breakeven within `+0.5% APR` of deployed (i.e., fails at ≤ 0.75% APR) |

A pair is **FRAGILE** if EITHER axis breaks within those bumps. A fragile pair stays deployed but **must not be sized up**, and live cost realization (real fill slippage, actual broker borrow rates) needs continuous monitoring.

---

## How to run

Per-pair invocation. Run A1-surviving pairs only:

```bash
# JPM/BAC (deployed):
python tools/a3_cost_stress.py \
    --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \
    --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \
    --train-days 30 --test-days 10 \
    --pair-entry-z 1.5 --pair-exit-z 0.4 \
    --workers 8 \
    --output a3_jpm_bac.csv

python tools/a3_analyze.py --input a3_jpm_bac.csv
```

If you want A3 on GOOG/GOOGL anyway (it's currently demoted per A1, but the cost data is still informative for any future re-validation):

```bash
python tools/a3_cost_stress.py \
    --csv-y data/alpaca/goog_730d_1m.csv --symbol-y GOOG \
    --csv-x data/alpaca/googl_730d_1m.csv --symbol-x GOOGL \
    --train-days 30 --test-days 10 \
    --pair-entry-z 1.5 --pair-exit-z 0.4 \
    --workers 8 \
    --output a3_goog_googl.csv
python tools/a3_analyze.py --input a3_goog_googl.csv
```

The analyzer accepts a multi-pair CSV: `cat a3_jpm_bac.csv a3_goog_googl.csv > a3_basket.csv && python tools/a3_analyze.py --input a3_basket.csv` (handle the duplicate header manually or just keep them separate).

`--resume` works the same way as in A2: re-running with the same `--output` path skips cells already present.

---

## Interpreting the output

Sample analyzer output (real data will replace the example numbers):

```
A3 cost-stress surface for JPM/BAC
  cells=9  alpha=0.05  Bonferroni per-cell threshold=0.00556

          borrow:     0.25%     1.00%     3.00%
-----------------------------------------------
  slip  1.5bps  | +0.441B* +0.412B  +0.298r
  slip  3.0bps  | +0.318B  +0.289B  +0.176.
  slip  5.0bps  | +0.135.  +0.108.  -0.012.

  Deployed cell: mean=+0.441%  CI=[+0.21%, +0.82%]  raw_p=0.0070  edge+=True

  Breakeven boundaries:
    slippage : EDGE+ fails at 5.0 bps/side (+3.5 bps above deployed)
    borrow   : EDGE+ fails at 3.00% APR (+2.75% above deployed)

  VERDICT: JPM/BAC has comfortable cost margin at deployed assumptions.
```

`B` = passes Bonferroni; `r` = raw p < 0.05 only (no family correction); `.` = fail. `*` = deployed cost cell.

In the above hypothetical, JPM/BAC would clear A3 — breakeven is +3.5 bps/+2.75%, well above the +1.0 bps / +0.5% fragility thresholds.

---

## What A3's verdict determines

| A3 verdict (for any A1-surviving pair) | Action |
|---|---|
| **Comfortable margin** | Pair stays deployed at current size; proceed to **A4** (regime conditioning). |
| **FRAGILE** | Pair stays deployed but at REDUCED `target_dollar_notional` (recommend cutting by half), monitoring intensifies, and re-validation cadence increases. |
| **Deployed cell fails** | DEMOTE. The pair doesn't survive even at the deployed cost level — this overrides A1's marginal pass. Pause `main_pairs.py`. |

---

## What A3 does NOT do

- Does not re-tune strategy parameters. A2 covers that question.
- Does not change the cost model on master. The cost assumptions in `backtest_runner.TradeLedger` are unchanged; A3 only varies the values passed into a per-cell `BacktestConfig`.
- Does not test live slippage. A3 stress-tests assumptions; real-world fill data after 30 days of paper trading is the empirical follow-up.
