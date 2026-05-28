# F1 — Pre-Registration: Intraday Index-ETF Momentum

**STATUS: DRAFT (unfrozen).** This document is open for iteration. Freeze it
(change status to FROZEN, fill commit hash + date) **before the first F1
backtest touches data**. Once frozen, every researcher degree of freedom — symbols,
trigger math, parameter grid, window scheme, significance bar — must be fixed so
that no choice can be made after seeing results. Editing a frozen spec is a
protocol violation; the right path is a new F1b pre-registration with its own
commit and its own family-size accounting.

Commit hash at freeze time: ________________ (fill on commit)
Date frozen: ________________
Author: Adrian

---

## 0. One-line hypothesis

Intraday momentum on liquid index ETFs — opening a trend-following position
when intraday price action shows an abnormal directional imbalance off a
reference level, managed with a volatility-scaled trailing stop and force-flat
by EOD — produces a positive, cost-aware, out-of-sample edge that survives the
pre-registered family-wise-corrected significance bar.

Prior (not verdict): Zarattini, Aziz & Barbon (SSRN 2024) reported ~19.6%
annualized / Sharpe ~1.33 net of costs on SPY, 2007–early-2024. That is their
result on their harness and sets only the prior. It does not count as evidence
for deployment. F1 clears this harness or it does not deploy.

Disconfirming result is a success. If F1 comes back EDGE− or inconclusive at
the corrected bar, the correct outcome is to record that and NOT deploy —
exactly as ORB/VWAP (p=0.995 EDGE−) were correctly rejected.

---

## 1. Symbols

### 1.1 Primary universe — unleveraged liquid index ETFs

| Symbol | Underlying | Role |
|---|---|---|
| SPY | S&P 500 | Primary — the Zarattini reference instrument |
| QQQ | NASDAQ-100 | Family member — tech-tilted index |
| IWM | Russell 2000 | Family member — small-cap, higher intraday vol |

### 1.2 Leveraged variant — SEPARATE pre-registration sub-family

TQQQ (3x QQQ) and SPXL (3x S&P 500) are included only as an explicitly-flagged
sub-family, NOT pooled with the unleveraged names, because:

- Daily leverage reset causes path-dependent decay — these are intraday/overnight
  instruments only, never holds, and the buy-and-hold null (§6) must use the
  leveraged series, not the underlying index.
- Borrow/financing characteristics differ; the cost-stress (§5) must use
  leveraged-appropriate assumptions, not the equity-pair defaults.
- There is no 3x sector basket and none of these is a thematic/illiquid product —
  that was checked and rejected as a universe; only deep-liquidity index proxies
  qualify for F1.

| Symbol | Underlying | Notes |
|---|---|---|
| TQQQ | 3x NASDAQ-100 | decay-prone; intraday only; null = TQQQ B&H, not QQQ |
| SPXL | 3x S&P 500 | decay-prone; intraday only; null = SPXL B&H |

**Pooling rule:** unleveraged {SPY, QQQ, IWM} pool together for a pooled verdict.
Leveraged {TQQQ, SPXL} pool separately. A pooled EDGE+ on one sub-family does
NOT transfer to the other.

**Open spec question (resolve before freeze):** target dollar notional for the
leveraged sub-family. The code currently treats `target_dollar_notional`
identically for both sub-families. A $10k notional on TQQQ is 3× the underlying
exposure of $10k on QQQ, so apples-to-apples comparison needs either (a) a
sub-family-specific notional (e.g., $3,333 for leveraged) or (b) an explicit
note that the leveraged sub-family is measured at 3× underlying exposure. Pick
one and write it into the freeze.

---

## 2. Trigger definition

All quantities computed on the same 1-minute bar cadence the harness already
uses (via the `_bar_to_ticks` / TickResampler path), so live and backtest
cadence match. No intrabar lookahead: a signal computed from bar t may only be
acted on at the open of bar t+1.

### 2.1 Reference level (the "expected" intraday path)

For each session, define the volatility-scaled boundary off the regular-session
open:

- `open_px` = first 1-min bar open of the regular session (09:30 ET).
- `sigma_intraday` = rolling estimate of the session's typical intraday move,
  computed as the trailing N_vol-day mean of `(daily high − daily low) / open`
  using REGULAR-HOURS bars only (09:30–16:00 ET). Pre-market and after-hours
  bars are excluded so sigma reflects session-anchored range, not extended-hours
  noise. Frozen choice: trailing window excludes the current day (no lookahead).
  N_vol is a grid parameter (§3).
- Upper boundary at time t: `UB_t = open_px * (1 + k * sigma_intraday * f(t))`
- Lower boundary at time t: `LB_t = open_px * (1 - k * sigma_intraday * f(t))`
- `f(t)` = `sqrt(elapsed_fraction)` so the band widens with elapsed session
  fraction. The multiplier `k` is a grid parameter.

### 2.2 Entry (abnormal directional imbalance)

- Long when price closes a 1-min bar above `UB_t`.
- Short when price closes a 1-min bar below `LB_t`.
- One position per symbol per session. No pyramiding. No re-entry after a
  stop-out in the same session.
- No new entries after `no_entry_after` ET (grid parameter).

### 2.3 Exit (volatility-scaled trailing stop + hard EOD flat)

- On entry, set a trailing stop at `trail_mult * sigma_intraday` (in price
  terms) from the high-water mark (longs) / low-water mark (shorts).
  `trail_mult` is a grid parameter.
- Trailing stop ratchets only in the favorable direction; never loosens.
- Stop is evaluated against the bar's intrabar extreme (low for longs, high for
  shorts), not the bar close — intrabar pierce counts as a fill.
- Hard EOD liquidation at 15:55 ET via the existing EODLiquidationManager. The
  strategy carries a local EOD guard as a safety net; the manager is the
  mechanism.
- Optional protective hard stop at `nominal_stop_pct` as a disaster backstop
  (frozen at repo default; NOT a grid parameter).

### 2.4 Fill convention (frozen)

**Frozen choice:** signal computed on bar t close → fill at bar t+1 open. This
is the conservative, lookahead-free convention. The alternative (fill at t
close) is explicitly rejected so it cannot be silently swapped in later to
improve results.

---

## 3. Parameter grid (defines family size)

Every cell is one test. Family size is fixed here before the run.

| Parameter | Frozen values | Count |
|---|---|---|
| k (band width multiplier) | {1.0, 1.5, 2.0} | 3 |
| trail_mult (trailing stop, ×sigma) | {1.5, 2.5} | 2 |
| N_vol (vol lookback, days) | {14, 20} | 2 |
| no_entry_after (ET) | {14:00} | 1 |

Combinations per symbol = 3 × 2 × 2 × 1 = **12**.

### 3.1 Family-size accounting

- Unleveraged sub-family: 3 symbols × 12 combos = 36 individual tests, BUT the
  verdict is taken on the pooled cross-symbol result per parameter combo →
  **12 pooled tests**.
- Leveraged sub-family: 2 symbols × 12 combos, pooled → **12 pooled tests**.
- **Total decision family = 24 pooled tests.**

We deploy a parameter set, not a symbol-specific fit, so "pooled" is the honest
unit of analysis. **If any per-symbol result is ever invoked to justify
deployment, the family reverts to 60 (5 symbols × 12) and the bar must be
recomputed at family=60 before claiming anything.**

---

## 4. Walk-forward window scheme

Reuse the existing harness — no new windowing logic:

- Train: 30 days. Test: 10 days. Slide forward by `test_days` (10). No overlap.
- Run via `walkforward_basket.py` (pooled cross-symbol),
  `--strategy intraday_index_momentum`.
- Analyze via `analyze_walkforward.py`: bootstrap 95% CI (10k resamples) on mean
  OOS per-window return, Newey-West HAC standard errors with lag = ⌊n^(1/4)⌋.
- Data: same 2yr Alpaca 1-minute source as the pairs validation, same cost-model
  path.
- Walk-forward is the sole judge. Every grid cell goes through full walk-forward.

---

## 5. Cost model & stress

Directional trades carry net exposure, so costs are evaluated up front:

- Base: existing harness cost model (1.5 bps/side slippage, SEC fee, 0 commission).
- Stress grid (mandatory at first evaluation): slippage ∈ {1.5, 3, 5} bps/side.
- Leveraged sub-family additional: model the daily-reset financing drag and any
  borrow on shorts; the buy-and-hold null uses the leveraged series itself.
- Report breakeven slippage per sub-family. A config that is EDGE+ at 1.5 bps
  but EDGE− at 3 bps is flagged fragile and does not deploy.

---

## 6. Null benchmarks (the bar is higher than zero)

A directional intraday strategy must beat MORE than zero:

- Buy-and-hold null of the same instrument over the same windows (leveraged
  null uses the leveraged series).
- Random-entry null: same number of trades, same holding-period distribution,
  random entry times within session. Bootstrap the random-entry PnL
  distribution (≥1k resamples); F1's OOS mean must exceed the random-entry
  distribution's upper region, not merely exceed zero.

**Frozen deployment gate:** a config deploys only if it is EDGE+ at the
corrected bar (§7) **AND** beats buy-and-hold **AND** beats the random-entry
null. All three.

---

## 7. Pre-computed corrected significance bar

Decision family = 24 pooled tests (§3.1).

### 7.1 Bonferroni (primary, conservative)

- Per-test threshold = 0.05 / 24 = **0.00208**.
- A pooled config is EDGE+ only if `p(mean>0) < 0.00208` AND bootstrap 95% CI
  lower bound > 0.

This is more stringent than the JPM/BAC `p=0.007` that justified the flagship
pairs deployment. Directional new alpha should clear a high bar.

### 7.2 Benjamini-Hochberg FDR (secondary, reported alongside)

- Rank the 24 pooled p-values ascending; BH threshold at rank i = (i/24) × q,
  with q = 0.05.
- Report which configs pass BH as the less-conservative view, but Bonferroni
  is the deployment gate. BH is informational only.

### 7.3 What "pass" produces

- Bonferroni-pass + CI>0 + beats both nulls (§6) + not cost-fragile (§5) →
  candidate for paper.
- Anything else → recorded as EDGE− / inconclusive, NOT deployed.

---

## 8. Pre-committed decision table

Fill **after** the run; structure frozen **now**.

| Sub-family | Best pooled config (k, trail, N_vol) | OOS mean % | 95% CI lo | p(>0) | Bonf pass (<0.00208)? | Beats B&H? | Beats random? | Cost-fragile? | VERDICT |
|---|---|---|---|---|---|---|---|---|---|
| Unleveraged (SPY/QQQ/IWM) | | | | | | | | | |
| Leveraged (TQQQ/SPXL) | | | | | | | | | |

Only the pre-specified "best pooled config" per sub-family — defined as
**lowest pooled p-value** — is eligible for the verdict. We do NOT scan all 12
and pick the prettiest equity curve.

---

## 9. Anti-gaming rules

1. No edits to this file after freeze — changes require a new F1b pre-registration.
2. No parameter values outside §3. No new symbols outside §1. No new exit logic
   outside §2.3.
3. The "best config" selection rule (§8: lowest pooled p) is fixed; no
   switching to "best Sharpe" or "best return" after seeing results.
4. Family size is 24 for the deploy decision; if any per-symbol result is
   invoked to justify deployment, recompute the bar at family=60 before
   claiming anything.
5. New alpha lands in `strategies/momentum/intraday_index_momentum.py` and is
   wired ONLY into the harness. `main_pairs.py` and `kalman_spread.py` are
   untouched.
6. A surviving config still requires 30 days clean paper before any
   live-capital discussion, and a §F3 correlation check against the deployed
   pairs basket before joining a combined book.
7. Disconfirmation is a valid, expected, fully-acceptable outcome.

---

## 10. Prerequisites (do not freeze F1 until both true)

- [ ] Workstream A is green (existing edge locked, multiple-comparisons audit done).
- [ ] B3 has run at least once (proves the harness cleanly ingests and scores
      a new candidate end-to-end, so an F1 anomaly can be attributed to the
      strategy, not the plumbing).
