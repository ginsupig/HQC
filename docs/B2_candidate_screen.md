# B2 — Cointegration + half-life pre-filter on B1 candidates

**Roadmap reference:** Workstream B, task B2. Cheap-but-discriminating second
screen in the pair-discovery funnel. Drops pairs whose y-on-x residual is not
stationary (Engle-Granger fails) or mean-reverts too slowly to trade.
Surviving pairs feed B3 (batch walk-forward).

---

## One-time install (research-tooling only)

`tools/b2_candidate_screen.py` imports `statsmodels.tsa.stattools.coint`.
This is intentionally NOT in `requirements.txt` because it is research
tooling and the deployed runtime does not need it. Install it once into
the HQC venv before running B2 for the first time:

```powershell
# Windows
.venv\Scripts\pip install "statsmodels>=0.14"
```

```bash
# Linux / macOS
.venv/bin/pip install "statsmodels>=0.14"
```

---

## Pipeline position

```
B1 (correlation screen)            → b1_*.csv     (candidate pairs)
B2 (cointegration + Hurst)         → b2_*.csv     (screen-pass survivors)
B3 (walk-forward harness)          → wf_*.json    (per-pair OOS PnL distributions)
B4 (pre-register corrected bar)    → docs/B4_*.md (family-size locked BEFORE results)
A1 (multiple-comparisons audit)    → verdict      (deploy / demote)
```

## Statistical rules (frozen for the screen)

| Leg | Test | Pass criterion | Default threshold |
|---|---|---|---|
| Cointegration | Engle-Granger via `statsmodels.tsa.stattools.coint(log(y), log(x), trend='c')` | `p < coint_p_max` | 0.05 |
| Tradability | OU half-life from residual: `Δresid_t = -θ * resid_{t-1} + ε`, half-life = ln(2)/θ | `half_life ≤ max_half_life_days` | 60 |
| (diagnostic) | Hurst exponent of residual via rescaled-range | reported only, **not a gate** | — |

A pair is `screen_pass = 1` only if **both gates** (cointegration AND
tradability) pass.

### Why Hurst is reported but not gated

Cointegration p-value already tests residual stationarity. Half-life captures
reversion speed. Gating on Hurst on top adds noise without adding signal —
the R/S estimator is small-sample-biased and its scaling convention is
sensitive to whether the input is treated as levels or increments. We
report Hurst because it's useful eyeballing alongside the two gated metrics,
but a pair is not rejected on Hurst alone.

`fail_reason` lists every failing gated leg, semicolon-separated.

## What B2 does NOT do

- **No PnL claim.** `screen_pass = 1` means "worth running B3 on," not
  "edge confirmed."
- **No significance correction.** B2's cointegration p-value is NOT the
  deployment p-value. A1 still owes a family-corrected p-value at
  family = K survivors of this screen.
- **No data fetching.** Missing CSVs are recorded with `fail_reason = missing_csv`
  so the pair shows up in the output but is clearly not scored. Use
  `python fetch_alpaca.py` to backfill.

---

## How to run

```bash
# Standard run on B1 financials output
python tools/b2_candidate_screen.py \
    --input b1_financials.csv \
    --output b2_financials.csv

# Tighter screen (3% cointegration p-value, 30d max half-life)
python tools/b2_candidate_screen.py \
    --input b1_financials.csv \
    --coint-p-max 0.03 \
    --max-half-life-days 30 \
    --output b2_financials_tight.csv

# Only score pairs B1 already flagged deployment_eligible=1
python tools/b2_candidate_screen.py \
    --input b1_financials.csv \
    --eligible-only \
    --output b2_financials_eligible.csv
```

Outputs:

| column | meaning |
|---|---|
| `y`, `x`, `sector` | Carried over from B1. |
| `n_overlap_days` | Trading-day intersection on which the tests were run. |
| `coint_t`, `coint_p` | Engle-Granger test statistic and MacKinnon p-value. |
| `alpha`, `beta` | OLS coefficients of `log(y) = alpha + beta*log(x) + eps`. |
| `hurst` | Hurst exponent of the residual. |
| `half_life_days` | OU mean-reversion half-life (days). |
| `screen_pass` | `1` if all three legs pass, else `0`. |
| `fail_reason` | Empty if `screen_pass=1`; otherwise semicolon list of failing legs. |
| `as_of`, `universe`, `b1_*` | Provenance carried from B1 for reproducibility. |

---

## Anti-look-ahead

`--as-of` defaults to the `as_of` column from the B1 input so a B2 re-run
sees the same data window. Set `--as-of` explicitly to re-screen at a
different date, but never use a date later than the B1 input's as-of without
also re-running B1 first; otherwise you'd be using B1's stale correlation
ranking with newer cointegration scores.

---

## Family-size accounting

If B1 → B2 → B3 evaluates K survivors end-to-end, A1's corrected significance
bar uses **family size = K**. The B2 screen does NOT count as multiple-testing
correction — failing the screen REMOVES a pair from the family; passing the
screen keeps it IN the family at the corrected bar.

That is B4's job: lock the corrected threshold for family = K **before** B3
results are seen. B2 should be re-run with frozen thresholds so K doesn't
drift after the bar is set.

---

## Picking the thresholds

| Threshold | Default | Reasoning |
|---|---|---|
| `coint_p_max` | 0.05 | Standard. Drop to 0.01 if K is large (>100) and you want a sharper screen before B3 burns walk-forward time. |
| `max_half_life_days` | 60 | A pair with 90-day half-life takes 3 months to revert, which won't fit cleanly into a 10-day walk-forward test window. 30 is tighter, 90 is looser. |

All thresholds are pre-screen knobs, not significance knobs. Moving them
changes which pairs reach B3, not whether a B3 survivor deploys.

---

## Smoke test

```bash
# Sanity: the deployed pair should pass cleanly
python tools/b1_universe_clustering.py --universe deployed --output /tmp/b1_d.csv
python tools/b2_candidate_screen.py --input /tmp/b1_d.csv --output /tmp/b2_d.csv
cat /tmp/b2_d.csv
```

Expected: one row, JPM-BAC, `screen_pass = 1` with `coint_p < 0.05` and
`half_life_days` somewhere in the 5–40 range. If the deployed pair fails B2,
something's wrong with the data or the thresholds — investigate before
trusting any other B2 output.
