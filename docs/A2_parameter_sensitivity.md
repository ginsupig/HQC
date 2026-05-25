# A2 — Parameter sensitivity / overfitting surface

**Roadmap reference:** Workstream A, task A2. Gating prerequisite for any new pair work (B). The deployed Kalman pairs runtime is not touched; A2 produces evidence about whether JPM/BAC's marginal A1 survival is robust to parameter perturbation or a knife-edge artifact.

**Standing-rule recap:** Significance bar requires bootstrap 95% CI lower bound > 0 AND family-corrected p < 0.05. A1 demoted GOOG/GOOGL and left JPM/BAC marginal (raw p=0.007, passes strict-family Bonferroni by 0.0013). A2 asks: if we perturb (entry_z, exit_z, δ, Ve), does the edge persist or collapse?

---

## What A2 measures

For each cell in the 4-D parameter grid, run the full validated walk-forward (47 OOS windows on 2 years of JPM/BAC 1-minute data) and record:

- OOS mean per-window return
- Bootstrap 95% CI on the mean
- Newey-West HAC p-value for `H1: mean > 0`
- Windows positive, windows with PF > 1, average PF
- Total OOS PnL and total costs

Default grid (from the roadmap):

| Parameter | Values | Count |
|---|---|---|
| `entry_z` | 1.25, 1.5, 1.75, 2.0 | 4 |
| `exit_z` | 0.2, 0.4, 0.6 | 3 |
| `delta` | 1e-5, 1e-4, 1e-3 | 3 |
| `ve` | 1e-3, 1e-2 | 2 |
| **non-degenerate cells (exit_z < entry_z)** | | **72** |

Each cell is independent walk-forward of 47 windows → ~3,400 backtests total.

---

## Verdict rules

A cell counts as **EDGE+** only when BOTH:

1. Bootstrap 95% CI lower bound > 0
2. Raw p-value ≤ Bonferroni threshold = `α / family_size`, where `family_size` = number of grid cells (default 72 → 0.000694)

The analyzer also reports Bonferroni-and-BH-FDR variants per cell so the operator can see graduated strictness.

**Plateau test:** Around the deployed `(entry_z=1.5, exit_z=0.4, δ=1e-4, Ve=1e-3)` cell, compute the **4-neighbour connected region** of EDGE+ cells (within the deployed `(δ, Ve)` slice). 

| Region size | Verdict |
|---|---|
| 0 (deployed cell fails Bonferroni) | **DEMOTE JPM/BAC** — surviving edge not present even at deployed parameters |
| 1 (only the deployed cell) | **SPIKE** — curve-fit; DEMOTE or re-validate on more data |
| ≥ 4 connected EDGE+ cells | **PLATEAU** — edge robust to parameter perturbation; JPM/BAC stays deployed |

---

## How to run

Sandbox doesn't have your 2-year Alpaca CSVs; this runs on your machine.

### Smoke test first (~10–20 min)

```bash
python tools/a2_parameter_sensitivity.py \
    --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \
    --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \
    --train-days 30 --test-days 10 \
    --smoke --workers 4 \
    --output a2_jpm_bac_smoke.csv

python tools/a2_analyze.py --input a2_jpm_bac_smoke.csv
```

The smoke grid is `entry_z ∈ {1.5, 2.0}, exit_z ∈ {0.2, 0.4}, delta=1e-4, ve ∈ {1e-3, 1e-2}` = 8 cells. Confirms the runner, the analyzer, and your machine's per-cell wall time before committing to the full overnight run.

### Full grid (overnight)

```bash
python tools/a2_parameter_sensitivity.py \
    --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \
    --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \
    --train-days 30 --test-days 10 \
    --workers 8 \
    --output a2_jpm_bac_full.csv
```

Wall-time estimate: ~13 s/window × 47 windows = ~10 min/cell. 72 cells / 8 cores ≈ 90 min ideal; in practice budget 2-4 hours. If interrupted, re-run with `--resume`.

```bash
python tools/a2_analyze.py --input a2_jpm_bac_full.csv \
    --deployed-entry-z 1.5 --deployed-exit-z 0.4 \
    --deployed-delta 1e-4 --deployed-ve 1e-3
```

---

## Interpreting the output

Sample analyzer slice (synthetic smoke run):

```
=== slice delta=1e-04  ve=1e-03 ===
  (deployed marker `*` at entry_z=1.5, exit_z=0.4)
entry_z \ exit_z      0.20      0.40
-----------------------------------
          1.50  +0.012.  +0.011.*
          2.00  +0.000.  +0.000.
  Legend: B=Bonferroni, F=BH-FDR only, r=raw p<0.05 only, .=fail
```

Each cell: `mean_pct  edge_mark  deployed_marker`. Walk the deployed-cell row and column and ask: are neighbours also `B`? If yes → plateau. If isolated → spike.

The OVERALL VERDICT block at the end of the analyzer output gives the binary call.

---

## What happens after A2

| A2 verdict for JPM/BAC | Action |
|---|---|
| **PLATEAU** | JPM/BAC stays deployed. A1's strict-family Bonferroni pass is upheld. Proceed to **A3** (cost stress) and **A4** (regime conditioning). |
| **SPIKE** | DEMOTE JPM/BAC. The inclusive-family A1 fail becomes the truth, and we have no validated pair. Workstream B (pair discovery) becomes urgent: we need new candidates with their own pre-registered corrected bar. |
| **DEMOTE** (deployed cell fails) | Same as SPIKE — JPM/BAC demoted. |

In SPIKE or DEMOTE cases, the operator should also pause `main_pairs.py` and run with no pairs (or `simulate_only=1`) while B proceeds.

---

## What A2 does NOT do

- Does not re-fit the Kalman filter parameters to maximize OOS edge. That would be exactly the overfitting the test is supposed to detect.
- Does not modify `main_pairs.py`, `kalman_spread.py`, or the harness. New work is in `tools/a2_*.py` only.
- Does not apply correction across pairs (that was A1); the family here is the parameter grid.
- Does not test the resampler / live-vs-backtest cadence question (settled in PR #4: byte-parity verified).
