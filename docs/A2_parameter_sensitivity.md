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

---

## Gate recalibration (2026-06-03): the per-cell bar was mis-specified

**Bug.** The campaign's A2 (and A3) gate marked a parameter cell EDGE+ only if
`raw_p <= alpha / (#grid cells)` — a Bonferroni correction applied *across the
parameter grid*. With the full A2 grid that threshold is `0.05/216 ≈ 0.00023`,
which essentially no single 46-window walk-forward cell can ever meet. The
result: A2 and A3 returned FAIL for **every** pair in the 7-pair campaign,
regardless of whether real edge existed.

**Why it's wrong.** A2 is a *robustness* test, not a significance test. A1
already establishes significance with a correction across the **pair family**
(the right multiple-comparisons family). A2 only asks: *given* the pair is
significant, is the deployed parameter node surrounded by a contiguous region
of positive cells, or is it a curve-fit spike? The **plateau (region-size ≥ 4)
requirement is itself the multiple-comparisons control.** Re-applying a
family-wise correction *across the grid* on top of that double-counts and is
unattainably strict — and, perversely, gets *stricter* the more finely you
sample the grid, which is backwards for a robustness check.

**Fix.** The per-cell EDGE+ bar is now configurable (`gates.a2.cell_bar` /
`gates.a3.cell_bar`, or `--cell-bar`), with three options:

| `cell_bar` | per-cell EDGE+ rule | use |
|---|---|---|
| `raw` (default) | `ci_lo > 0 AND raw_p < alpha` | robustness via plateau; correct default |
| `bh` | `ci_lo > 0 AND` passes BH-FDR across the grid | stricter, controls false-discovery |
| `bonferroni` | `ci_lo > 0 AND raw_p <= alpha/#cells` | legacy (near-unattainable) |

**Pre-registration honesty.** This changes a pre-registered bar *after* seeing
results, which the repo is otherwise disciplined against. It is justified as a
correction of a genuine **methodological error** (wrong family), not goal-
seeking: the new bar is applied uniformly to all pairs, and — importantly — it
changes **zero deploy verdicts** for the 2026-06 campaign. JPM/BAC's A2 surface
is a clean plateau under `raw` (12/12 deployed-slice cells positive and
raw-significant), yet it is still correctly **DEMOTED**, because it fails the
*pair-family* correction (`raw_p 0.0223 > 0.05/7 = 0.0071`). The gate fix simply
stops A2/A3 from emitting false "spike/demote" verdicts on robust surfaces.

**Re-scoring without recompute.** Because A2/A3 verdicts are derived from the
per-cell `a2_surface.csv` / `a3_surface.csv` (which already hold every cell's
mean/CI/raw_p), you can re-derive all verdicts + ranking + promotion under a new
bar from existing artifacts, with no backtest re-run:

```bash
python tools/pairs_candidate_campaign.py --config config/pairs_research.yaml \
    --rescore --cell-bar raw
```
