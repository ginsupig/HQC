# HQC — Hybrid Quantitative Trading System

An event-driven backtest + live deployment framework for US-equity intraday strategies, with rigorous walk-forward validation and one statistically-validated trading strategy currently deployed on Alpaca paper.

This README is a faithful record of what the system does, what it has been proven to do, and — equally important — what has been disproven. Every quantitative claim below is backed by a specific walk-forward result in the repo.

---

## Quick verdict

Significance bar updated 2026-05-25 (A1 audit, `docs/A1_multiple_comparisons.md`): bootstrap 95% CI lower bound > 0 AND family-wise / FDR-corrected p < 0.05. Single-test p < 0.05 is **no longer sufficient**.

| Strategy | Wired? | Raw p | Corrected verdict (family) | Deployed? |
|---|---|---|---|---|
| **Kalman pairs — JPM/BAC** | yes | 0.007 | Survives Bonferroni at strict family n=6 by 0.0013 — **MARGINAL**; A2 gating | **yes (sole pair)** |
| Kalman pairs — GOOG/GOOGL | yes | 0.031 | Fails Bonferroni and BH-FDR at any reasonable family — **DEMOTED** | no (config-commented) |
| ORB (Opening Range Breakout) | yes | n/a | **EDGE−** pooled p=0.995 (loses) | no — guarded |
| VWAP Hunter | yes | n/a | **EDGE−** pooled p=0.995 (loses) | no — guarded |
| ORB + VWAP + ML gate | yes | n/a | no lift, IR −1.6 | no — guarded |
| Overnight gap-fade | yes | n/a | no edge, p=0.901 | no |
| KO/PEP pairs | yes | 0.142 (best) | no edge | no |
| XOM/CVX pairs | yes | 0.143 | no edge | no |
| AMD/NVDA pairs | yes | ≈0.14 | no edge | no |
| AAPL/META pairs | yes | small-n inconclusive | not actioned | no |

**Deployed basket is now a single pair, JPM/BAC, on marginal post-correction significance.** A2 (parameter sensitivity) is the gating prerequisite to confirm the edge is a plateau and not a knife-edge artifact at exactly (entry_z=1.5, exit_z=0.4). If A2 shows a plateau, JPM/BAC remains. If it shows a spike, the conservative-family interpretation (n=8) becomes the truth and JPM/BAC should also be demoted.

All non-deployed strategies are retained as harness baselines and as documented disproofs; `git` history is the proper archive for failed experiments.

---

## How to run the deployed system

```bash
# Set API keys (paper trading on Alpaca):
#   .env: ALPACA_API_KEY=...
#         ALPACA_API_SECRET=...

python main_pairs.py --config config/pairs.yaml
```

The runtime:
1. Loads pair definitions from `config/pairs.yaml`.
2. Connects to Alpaca's IEX websocket (live trades) and paper REST API (orders).
3. Resamples raw trades to 1-minute bar cadence (`TickResampler`) so the live Kalman filter sees the same tick stream cadence as the validated backtest.
4. Instantiates one `USEquityKalmanPairsTrader` per pair.
5. Runs a `PairsRiskMonitor` (beta-drift kill switch) and prints a heartbeat every 120s.

To go from paper to live capital requires **both** `paper: false` in the config **and** `HQC_ENABLE_LIVE_TRADING=1` in the environment — the runtime refuses to start otherwise. Do not bypass that gate without 30 days of clean paper-trading evidence.

The legacy `main.py` (ORB+VWAP) is **guarded**: it refuses to run for trading without `HQC_ALLOW_LEGACY_MAIN=1`, because both strategies it wires are statistically significant negative-edge.

---

## The deployed strategy: Kalman pairs

### Mechanic

Stat-arb on a cointegrated pair. Given two correlated symbols Y and X, model the relationship as `y = β·x + ε` where β is a slowly-varying hedge ratio. The system:

1. Estimates β online with a 1-D Kalman filter (`strategies/mean_reversion/kalman_spread.py:_update_filter`). Process noise `δ` controls how fast β can drift; measurement noise `Ve` controls how much innovation noise we accept. Defaults `δ=1e-4`, `Ve=1e-3`.
2. Each tick, computes the innovation `e_t = y_t − β·x_t` and standardizes it by the Kalman innovation variance: `z_t = e_t / √S_t`.
3. When `|z_t| ≥ entry_z` (default 1.5σ), opens a spread:
   - `z > 0`: spread is wide → short Y, long X
   - `z < 0`: spread is compressed → long Y, short X
4. When `|z_t| ≤ exit_z` (default 0.4σ), closes the spread.
5. Per-leg share counts are derived from a target dollar notional and the current β so the spread is dollar-balanced on entry (`USEquityKalmanPairsTrader._compute_entry_sizes`). The same share counts are re-used on exit so positions close to zero rather than leaving qty=1-3 residuals.
6. A configurable `cooldown_seconds` between signals (default 120s) and a `nominal_stop_pct` per-leg protective stop limit downside on regime breaks.

### Why JPM/BAC and GOOG/GOOGL

These two pairs were chosen after testing 7 candidate pairs (semis, consumer staples, energy, mega-cap tech, banks). They are the only two that survived walk-forward + bootstrap significance + Newey-West HAC standard errors on 2 years of Alpaca 1-minute data:

- **JPM/BAC** — same-sector banking duopoly, shared macro drivers (rates, credit spreads, regulation), strong cointegration relationship. β stable in the 5.8–6.0 range across 2 years.
  - **47 OOS windows, 41 positive PnL, avg PF 1.53, mean per-window return +0.441%, 95% CI [+0.21%, +0.82%], p = 0.007**
  - +$20.7k OOS PnL on $10k per-leg notional over 2 years (~10% CAGR market-neutral).

- **GOOG/GOOGL** — dual-class shares of the same company. Cointegration is structural (mechanical arbitrage between voting and non-voting). Trades infrequently but with high profit factor when it does (PF 8.2).
  - **47 windows, 6 positive PnL (rest mostly 0 trades — the spread rarely deviates by 1.5σ on a structural arb), mean +0.027%, CI [+0.004%, +0.061%], p = 0.031**
  - +$1.3k OOS PnL — small but real, and uncorrelated with JPM/BAC's bank exposure.

Both pairs have stable β across the 2-year window — no cointegration breakdowns detected. The deployed basket is the simplest meaningful diversifier: one financials pair + one mega-cap-tech pair, different mechanics, different sector exposure.

### Pairs that were tested and rejected

| Pair | OOS verdict | Notes |
|---|---|---|
| AMD/NVDA | p ≈ 0.14, borderline-no-edge | 47 windows, only 55% positive |
| KO/PEP | p = 0.142 at loose entry, no edge | Too tightly cointegrated; spread barely deviates |
| XOM/CVX | p = 0.189, no edge | Outlier-window dependent |
| AAPL/META | n = 10–15 | Inconclusive; need longer history |

These were not arbitrary picks — they are the textbook stat-arb pairs (consumer-staples duopoly, energy majors, mega-cap tech). The fact that 3 of them failed and 1 borderline-passed is the most honest result in the repo: published "textbook" pairs don't necessarily transfer to current markets. Bank pairs and dual-class arbitrage are the survivors.

---

## What is *not* deployed — and why

These strategies are wired into the codebase and the backtest harness, but they were rigorously disproven and the runtime that wires them (`main.py`) is guarded so it can't be run for trading by accident.

### ORB (Opening Range Breakout) — `strategies/orb/equity_orb.py`

Captures the 15-minute opening range, then trades breakouts above the range high or below the range low, with the protective stop placed a configurable buffer beyond the opposite end of the range.

**Walk-forward verdict on 7 symbols (SPY, AAPL, NVDA, TSLA, AMD, COIN, META), 47 OOS windows each, with 1.5 bps/side slippage and Alpaca's SEC fee:**

- Pooled mean OOS return per window: **−0.190%**
- Pooled 95% CI: **[−0.340%, −0.043%]** — entirely below zero
- **p(>0) = 0.995** — statistically significant *negative* edge

It is not a "no edge" strategy. It is a *negative*-edge strategy, confirmed at 99.5% confidence. Costs matter: PnL was -$8,101 without friction and -$11,974 with realistic costs.

### VWAP Hunter — `strategies/vwap/hunter_state_machine.py`

Volume-weighted-average-price bounce strategy: enters long when price tags VWAP from above after a confirmation tick.

Same walk-forward methodology, same verdict: pooled p=0.995 EDGE−.

### ORB + VWAP + LogReg ML gate — `intelligence/ml_pipeline/ml_candidate_gate.py`

Adds a deliberately conservative ML filter: PCA over 10 stationary technical features → L2-regularized class-balanced logistic regression → veto candidates whose probability disagrees with their direction. Fit on the leading 5000 bars of the dataset (no leakage; strategies trade only on the remainder).

Result on SPY rules: gate vetoed 95% of candidates (191 of 201). Of the 10 it let through, 60% won, but losses still bigger than wins. **IR = −1.63 vs ungated IR = −1.60.** The gate correctly identified that there was no learnable predictability subset; it didn't add lift because the underlying rules-based candidates have no exploitable signal to find.

This is itself an important result: a well-implemented ML filter is not magic. If the underlying signal has no edge, no model can manufacture one.

### Overnight gap-fade — `strategies/gap_fade/overnight_gap_fade.py`

Designed as a baseline validator: the published Boudoukh/Richardson/Whitelaw 2008 overnight-gap-fade anomaly is one of the few intraday effects with consistent academic support. If even *that* failed in our harness, the harness has a bug; if it works, we can trust the negative findings on ORB/VWAP.

**Result: 7 symbols, 47 OOS windows each (after fixing a ranker bug that was silently vetoing the gap-fade entries — see `intelligence/candidate_ranker.py`), p = 0.901, POOLED no edge.** COIN individually rejects with p = 0.997 EDGE− because crypto-equity gaps trend rather than fade — gap-fade is structurally backward there.

The interpretation: the published anomaly may have decayed in modern markets, or our 6-month window happened to be unfavorable, or the time-of-day filter parameters need tuning. The honest read is "not deployable as-is."

### Single-symbol Kalman pairs (AAPL/META, AMD/NVDA)

These were the *tech-pair* candidates. AMD/NVDA's 47-window result has avgPF 1.93 but mean per-window only +0.081% with a wide CI; rough significance check ≈ p=0.14. Not edge. AAPL/META has only n=10-15 — inconclusive.

---

## Architecture

```
                      ┌─────────────────────┐
                      │  Alpaca IEX WS feed │ (data/feeds/ws_manager.py)
                      └──────────┬──────────┘
                                 │ raw trade ticks (EventType.TICK)
                                 ▼
                      ┌─────────────────────┐
                      │   TickResampler     │ (data/feeds/tick_resampler.py)
                      │  raw → 1-min OHLC   │  buckets per symbol; on bar close
                      │  emits 4 ticks/bar  │  emits 4 BAR_TICK events
                      └──────────┬──────────┘
                                 │ EventType.BAR_TICK
                                 ▼
            ┌──────────────────────────────────┐
            │  USEquityKalmanPairsTrader (×N)  │  one per pair
            │  Kalman β + z-score → ORDER_CREATE│
            └──────────┬───────────────────────┘
                       │ stage=SIZED ORDER_CREATE (with explicit shares)
                       ▼
                ┌──────────────────┐
                │  Alpaca paper    │ (core/execution/broker_router.py)
                │  REST router     │
                └──────────────────┘

       ┌────────────────────────────┐
       │   PairsRiskMonitor         │  beta-drift kill switch
       │   subscribes to TICK + FILL│
       └────────────────────────────┘

       ┌────────────────────────────┐
       │   EODLiquidationManager    │  flatten any open positions at 15:55 ET
       └────────────────────────────┘
```

### Event bus

A single asyncio `EventBus` (`core/engine/event_bus.py`) routes events between components. Strategies are decoupled from the broker and data feed — they consume `TICK` / `BAR_TICK` and emit `ORDER_CREATE`; the broker consumes `ORDER_CREATE` and emits `ORDER_FILL`. Risk monitors subscribe to whatever they need.

Event types: `TICK`, `BAR_TICK`, `ORDER_CREATE`, `ORDER_FILL`, `EQUITY_UPDATE`, `SYSTEM_SHUTDOWN`, plus diagnostics.

### Order lifecycle

Strategies emit `ORDER_CREATE` with one of three stages:

- **(no stage)** — raw strategy intent. Used by ORB and VWAP. Goes through the rules-based `CandidateRanker` for microstructure scoring, then `DynamicRiskSizer`, then the broker.
- **`stage="RANKED"`** — pre-approved, skips the score gate (used by gap-fade since its conviction is the overnight gap itself, not microstructure).
- **`stage="SIZED"`** — fully pre-sized with explicit `shares` and `decision_id`. Skips both ranker and sizer (used by Kalman pairs, since the strategy must control beta-hedged sizing itself).

### TickResampler (the live-vs-backtest cadence fix)

The single most consequential infrastructure change in this codebase. `backtest_runner._bar_to_ticks` replays each 1-minute OHLCV bar as 4 synthetic ticks (O/L/H/C path, 15s-spaced). The Kalman filter advances its process-noise covariance once per tick, so in the backtest it advanced 4 steps per minute. The live Alpaca websocket emits every raw trade — 50-500+ per minute for liquid names — which would make the filter adapt ~100× faster than in the backtest, β would hug the instantaneous price ratio, innovations would collapse, and z-scores would never reach the entry threshold. The strategy would look alive but never trade.

`TickResampler` buckets raw trades into 1-minute OHLC bars per symbol and on each bar boundary emits 4 `BAR_TICK` events that are **byte-for-byte identical** to `_bar_to_ticks` output. The live filter advances at exactly 4 steps/minute — the validated cadence.

### Cost model

`backtest_runner.TradeLedger._round_trip_cost` applies, on every closed trade:

1. **Slippage** — 1.5 bps each side (3 bps round-trip), conservative for Alpaca-routed equity orders.
2. **Commission** — 0 per share (Alpaca's live equity rate), configurable.
3. **SEC fee** — 0.000008 of sell-side notional (current Alpaca live rate).
4. **Short-borrow fee** — 0.25% APR prorated by hold time, charged only on short legs. Default rate is reasonable for liquid large-caps (JPM, BAC, GOOG, GOOGL, NVDA, AMD). Hard-to-borrow names would require higher rates.

These are deliberately conservative. Total cost drag in backtest is typically 5-15% of gross PnL.

### Simulated exit engine (backtest only)

`backtest_runner._SimulatedExitEngine` enforces stop-loss brackets and a time-stop in the offline harness because `_simulate_fill` does not honor the `stop_loss_price` field in order payloads. Without this, the validated backtest results would be optimistic by ~50%. Live trading uses Alpaca's native OTO bracket, so this engine is not used in `main_pairs.py`.

---

## Validation methodology

Every "edge" claim in this README is supported by:

1. **Walk-forward analysis** with no train/test overlap. Default windows: 30-day train, 10-day test, sliding forward by `test_days`. For pairs: `walkforward_pairs.py`. For single-symbol strategies: `walkforward_basket.py`.

2. **Bootstrap 95% confidence intervals** on the mean OOS per-window return (10k resamples). Implemented in `analyze_walkforward.py`.

3. **Newey-West HAC standard errors** with lag = ⌊n^(1/4)⌋ to control for serial correlation across walk-forward windows when computing p-values.

4. **Pooled cross-symbol verdicts** when testing baskets (`walkforward_basket.py`), so single-symbol noise doesn't dominate.

5. **Strict gating: EDGE+ only if** the 95% CI lower bound strictly exceeds zero AND p(mean > 0) < 0.05.

A representative output (the JPM/BAC verdict that justifies deployment):

```
====================================================================
Walk-forward OOS return-per-window stats  (95% CI)
====================================================================
symbol      n   mean%    95%CI_lo   95%CI_hi   p(>0)   verdict
--------------------------------------------------------------------
JPM/BAC    47   0.441%   0.210%     0.817%     0.007   EDGE+
====================================================================
```

This is what a positive result looks like in this repo. Anything weaker than that is not deployed.

### Run the validation yourself

```bash
# Re-validate the deployed pairs (~10 min wall clock each):
python walkforward_pairs.py \
  --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \
  --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \
  --train-days 30 --test-days 10 \
  --pair-entry-z 1.5 --pair-exit-z 0.4 --pair-cooldown-seconds 120 \
  --sim-max-hold-minutes 240 --skip-train \
  --output result_wf_jpm_bac.json
python analyze_walkforward.py --input result_wf_jpm_bac.json

# Same for GOOG/GOOGL.

# Re-validate the negative-edge baselines (proves the harness can fail too):
python walkforward_basket.py --input data/alpaca --strategy both \
  --train-days 30 --test-days 10 --sim-max-hold-minutes 60 --summary-only \
  --output result_wf_rules.json
python analyze_walkforward.py --input result_wf_rules.json
```

You should get `JPM/BAC EDGE+ p=0.007`, `GOOG/GOOGL EDGE+ p=0.031`, and pooled-rules `EDGE− p=0.995`.

---

## Risk controls

### Pre-trade (in strategy)

- **Per-leg dollar notional cap** via `target_dollar_notional` (default $10k). Pairs never deploy more than ~$20k gross per spread on $100k account.
- **Per-leg protective stop** via `nominal_stop_pct` (default 2%). For pairs this is a backstop, not the primary exit — the primary exit is z-reversion.
- **Cooldown between signals** via `cooldown_seconds` (default 120s) to prevent same-tick re-entry.
- **Cross-leg freshness** — strategy skips evaluation if the two legs' timestamps differ by more than `max_leg_staleness_sec` (default 30s).

### Runtime (in `PairsRiskMonitor`)

- **Beta-drift kill switch** — halts a pair if its Kalman β drifts more than `beta_drift_pct_kill` (default 30%) from the rolling `beta_drift_window_min` baseline (default 60 minutes). This is the *primary* live safeguard, because cointegration breakdown is the actual failure mode for stat-arb. When β goes off the rails, the spread is no longer mean-reverting; better to halt and reassess than keep trading a broken model.
- **Daily PnL kill** — stubbed pending a tighter PnL feed. Beta-drift is more diagnostic; daily PnL kill is a convenience.

### Live-trading gate

The runtime refuses to start in live (real money) mode unless **both** `paper: false` in config **and** `HQC_ENABLE_LIVE_TRADING=1` in environment. Two gates so neither alone can flip a paper bot live by accident.

The legacy `main.py` (ORB + VWAP) has its own gate: requires `HQC_ALLOW_LEGACY_MAIN=1` to run at all, because the strategies it wires were proven to lose.

---

## Directory layout

```
HQC/
├── main_pairs.py                        ← deployment runtime (Kalman pairs)
├── main.py                              ← legacy runtime (ORB+VWAP) — guarded
├── config/
│   ├── pairs.yaml                       ← deployed pair basket
│   └── settings.yaml                    ← legacy system config
├── strategies/
│   ├── mean_reversion/kalman_spread.py  ← *the deployed strategy*
│   ├── orb/equity_orb.py                ← ORB (validated EDGE−)
│   ├── vwap/hunter_state_machine.py     ← VWAP (validated EDGE−)
│   └── gap_fade/overnight_gap_fade.py   ← gap-fade (no edge)
├── data/feeds/
│   ├── ws_manager.py                    ← Alpaca IEX websocket
│   └── tick_resampler.py                ← raw→bar cadence converter
├── core/
│   ├── engine/event_bus.py              ← asyncio event router
│   └── execution/broker_router.py       ← Alpaca paper/live REST router
├── intelligence/
│   ├── candidate_ranker.py              ← rules-based microstructure score
│   ├── liquidity_rs_engine.py           ← liq/RS/spread metrics
│   └── ml_pipeline/
│       ├── feature_engineering.py       ← PCA over technicals
│       └── ml_candidate_gate.py         ← LogReg veto (no lift demonstrated)
├── risk/
│   ├── position_sizing/confidence_scaler.py   ← DynamicRiskSizer (rules-path)
│   └── virtual_monitor/equity_slope_detector.py
├── backtest_runner.py                   ← single-shot backtest + cost model + sim-exit
├── portfolio_batch_runner.py            ← multi-symbol batch backtest
├── walkforward_runner.py                ← single-symbol walk-forward
├── walkforward_basket.py                ← multi-symbol walk-forward
├── walkforward_pairs.py                 ← pair-aware walk-forward
├── analyze_walkforward.py               ← bootstrap CI + Newey-West p-values
└── tests/                               ← 20 unit tests (all passing)
```

---

## What this codebase does *not* claim

- It does not claim ORB or VWAP work. They are wired into the harness as known-negative comparison baselines and explicitly proven to lose.
- It does not claim a machine-learning model adds alpha. The LogReg gate was tested and did not add lift; it is included to document the experiment.
- It does not claim every cointegrated pair works. 4 textbook pairs failed; 2 succeeded. Pair selection matters more than parameter tuning.
- It does not claim the live behavior will exactly match the backtest. The `TickResampler` matches cadence, but live-only effects (real fills, latency, IEX market-data gaps, hard-to-borrow events) cannot be backtested. That is why a 30-day paper validation period is mandated before any live capital.

---

## Pre-deployment checklist

| Item | Status |
|---|---|
| Walk-forward EDGE+ p<0.05 on each deployed pair | ✓ JPM/BAC p=0.007, GOOG/GOOGL p=0.031 |
| Leg-sizing verified (no qty residuals) | ✓ |
| Realistic cost model (slippage + SEC fee + borrow) | ✓ |
| Beta-drift kill switch | ✓ |
| Daily PnL kill switch | ◐ stubbed |
| TickResampler — live cadence = backtest cadence | ✓ byte-parity verified |
| 30 days paper trading with corrected feed | ← *next* |
| Live capital | only after the above |

---

## Origin

This codebase was a real research → deployment cycle. Every claim above was tested. The session that produced this README started with a request for a code audit; it ended with one validated pair-trading strategy deployed on paper. Along the way it also produced rigorous *disproofs* of three other strategies, which is at least half the value of the exercise. The proper test of a quant system isn't "what did it find" but "what did it correctly *reject*."

The validated edge is small. ~10% CAGR on a market-neutral pair trading $10k notional per leg is plausible, sustainable, and unimpressive in isolation. It is what *real* stat-arb edge looks like before leverage. Treat it accordingly.
