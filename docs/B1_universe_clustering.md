# B1 — Universe + within-sector clustering for pair discovery

**Roadmap reference:** Workstream B, task B1. Cheap pre-filter that produces a
ranked CSV of candidate (y, x) pairs for B2 (cointegration + Hurst screen) to
score in detail.

---

## What B1 does

1. Loads a curated universe from `config/b1_universes.yaml` — a hand-written
   list of tickers with GICS-style sector labels.
2. Reads each ticker's 1-minute CSV from `data/alpaca/`, derives daily closes
   from regular-session bars only (09:30–16:00 ET), and computes day-over-day
   log returns.
3. For every within-sector pair, computes the Pearson correlation of daily
   log returns on the intersection of dates.
4. Flags pairs as `deployment_eligible` when both:
   - `corr ≥ --min-corr` (default 0.55)
   - `n_overlap_days ≥ --min-overlap-days` (default 252)
5. Writes a CSV sorted with eligible pairs first, then by correlation desc.

## What B1 does NOT do

- **No cointegration test.** That's B2. Correlation is the cheap pre-filter
  because cointegration is expensive to run and even more expensive to
  multiple-comparisons-correct at scale.
- **No significance claim.** `deployment_eligible` means "passed the screen,"
  not "edge has been validated." Validation requires A1/A2/A3/A4/D3 on the
  surviving candidate.
- **No data fetching.** Missing tickers are reported and skipped. Use
  `python fetch_alpaca.py --symbols <comma-list>` to backfill.

---

## Anti-look-ahead

`--as-of` defaults to today's UTC date and only sessions **strictly before**
that date are used. A B1 run on the same `--as-of` produces the same candidate
list given the same input CSVs — re-runs reproduce.

Always set `--as-of` explicitly when re-running historical screens; otherwise
new data added since the original run will silently change the verdict.

---

## How to run

```bash
# Sanity check: the one deployed pair
python tools/b1_universe_clustering.py \
    --universe deployed \
    --output b1_deployed.csv

# Financials — the natural family expansion around JPM/BAC
python tools/b1_universe_clustering.py \
    --universe large_cap_financials \
    --as-of 2026-05-26 \
    --min-corr 0.55 \
    --min-overlap-days 252 \
    --output b1_financials.csv

# Sector SPDR ETFs — useful as an ETF-vs-ETF sanity benchmark
python tools/b1_universe_clustering.py \
    --universe sector_etfs \
    --output b1_etfs.csv
```

Outputs:

| column | meaning |
|---|---|
| `y`, `x` | Pair members, alphabetically ordered (`y < x`). |
| `sector` | Common sector. Cross-sector pairs are intentionally excluded. |
| `n_overlap_days` | Trading-day intersection used for correlation. |
| `corr` | Pearson correlation of daily log returns. |
| `deployment_eligible` | `1` if `corr ≥ min_corr` AND `n_overlap_days ≥ min_overlap_days`, else `0`. |
| `as_of` | The date threshold; sessions strictly before this were used. |
| `universe` | The universe name passed via `--universe`. |

---

## Picking the thresholds

- **`min_corr = 0.55`** is intentionally permissive. Cointegrated pairs are
  often only correlated in the 0.5–0.8 range on daily returns; pushing this
  to 0.8 prematurely drops genuine candidates. B2 will narrow further.
- **`min_overlap_days = 252`** ≈ one trading year. Pairs with less overlap
  produce noisy correlation estimates and shaky cointegration tests in B2.
  Bump to 504 (~2 years) for tighter screens.

Both are screen knobs, not significance knobs — moving them does NOT change a
deployment claim. Only A1's family-corrected p-value does that.

---

## Family-size accounting (read this before deploying anything from B's funnel)

If B1 → B2 → B3 → A1 evaluates K candidate pairs end-to-end, the corrected
significance bar in A1 must use **family size = K**, not 1. That's B4's job
(pre-register the bar for the new family before B3 results are seen).

The cheap screens (B1 + B2) shrink K so the bar stays clearable. They do NOT
count as multiple-testing corrections themselves — failing the screen
removes a pair from the family; surviving the screen still owes a corrected
p-value in A1.

---

## Adding a new universe

Edit `config/b1_universes.yaml`:

```yaml
my_universe:
  - { ticker: ABC, sector: My Sector }
  - { ticker: XYZ, sector: My Sector }
```

Then:

```bash
python fetch_alpaca.py --symbols ABC,XYZ --days 730
python tools/b1_universe_clustering.py --universe my_universe --output b1_my_universe.csv
```

No tool changes needed.
