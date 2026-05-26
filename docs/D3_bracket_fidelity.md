# D3 — Backtest vs live OTO-bracket fidelity audit

**Roadmap reference:** Workstream D, task D3. Measurement only — produces a number, no runtime changes. Establishes whether the validated backtest's exit semantics actually match what Alpaca's OTO bracket will do live.

---

## The gap being audited

`backtest_runner._SimulatedExitEngine` closes a pair position on the first of three triggers:

1. **Strategy-side close** — z-score reverts to `exit_z`, `kalman_spread.py` emits SELL/BUY_TO_COVER.
2. **Stop-loss** — price crosses `stop_loss_price` (the same level that gets attached to live Alpaca brackets).
3. **Hard time stop** — position open longer than `sim_max_hold_minutes` (default **240 min**).

Live trading via Alpaca uses native **OTO bracket** on the entry order. Only (1) and (2) fire live. There is **no time stop** on a live bracket — positions hold until z-reversion or the 15:55 ET EOD liquidator flattens everything.

If a meaningful share of validated-backtest trades close via trigger (3), the live realized PnL will differ from the backtest. D3 quantifies that gap.

---

## How D3 quantifies it

Run the same walk-forward TWICE on the same windows:

| Config | sim_max_hold_minutes | Interpretation |
|---|---|---|
| `with_ts` | 240 (validated default) | What the backtest actually does. |
| `without_ts` | 999999 (effective no-op) | What live OTO bracket actually does. |

Per window, diff `total_return_pct`. Report:

- Per-window delta mean
- Bootstrap 95% CI on the delta
- Relative gap = `(mean_delta / mean_with_ts) * 100%`

## Verdict thresholds

| Relative gap | Verdict | Action |
|---|---|---|
| < 10% | **FIDELITY GOOD** | Live behavior approximately matches validated backtest. |
| 10–30% | **FIDELITY ATTENTION** | Modest divergence expected live; budget for it. |
| > 30% (+) | **FIDELITY FLAG (optimistic)** | Time stop is concealing losses by closing trades early. **Live edge will be SMALLER** than backtest. Re-validate with `--sim-max-hold-minutes 999999` or implement a runtime time-stop in `main_pairs.py` to mirror the backtest. |
| > 30% (−) | **FIDELITY FLAG (pessimistic)** | Time stop is closing winners early. Live edge will be LARGER. Less worrisome but still a model error. |

The "FIDELITY GOOD" threshold is intentionally tight (<10%) because A1 left JPM/BAC at marginal Bonferroni significance — a 20% gap there could be the difference between a deployable pair and a non-deployable one.

---

## How to run

```bash
python tools/d3_bracket_fidelity.py \
    --csv-y data/alpaca/jpm_730d_1m.csv --symbol-y JPM \
    --csv-x data/alpaca/bac_730d_1m.csv --symbol-x BAC \
    --train-days 30 --test-days 10 \
    --pair-entry-z 1.5 --pair-exit-z 0.4 \
    --workers 4 \
    --output d3_jpm_bac.csv
```

The same script does both runs and prints the summary at the end. `--resume` works the same way as the A2/A3 tools: re-running with the same `--output` skips configs already present and re-summarizes from disk.

**Wall time:** 2 configs × 47 windows = 94 backtests per pair. ~10-15 min single-process, ~5 min with 4 workers.

---

## Interpreting the output

Sample synthetic output (zero divergence because z-reversion happens before the time stop on this synthetic pair):

```
D3 bracket-fidelity audit  AAA/BBB
  n_windows=5  alpha=0.05

  with_ts     (sim_max_hold=240m, validated config):
    mean per-window return = +0.0111%   total PnL = $+55.37   trades = 6
  without_ts  (sim_max_hold=999999m, live-bracket semantics):
    mean per-window return = +0.0111%   total PnL = $+55.37   trades = 6

  Per-window delta (with_ts - without_ts) = +0.0000%
    bootstrap 95% CI on delta: [+0.0000%, +0.0000%]
    Relative gap: time-stop adds +0.0% to the validated mean.

  VERDICT: FIDELITY GOOD (+0.0% gap).
```

On real JPM/BAC data the gap will likely be small but nonzero. The closer to 0 it is, the more confidence we have that the validated backtest's numbers translate to live.

---

## What D3 does NOT do

- Does not change `_SimulatedExitEngine` or the backtest harness. Pure measurement.
- Does not test stop-trigger price fidelity (the assumption that the backtest's stop-fill matches the live OTO bracket's fill price). That's modeled by the cost-model slippage and tested separately in A3.
- Does not measure the gap for non-pair strategies (gap-fade, ORB, VWAP). They're guarded; D3 is scoped to the deployed pair path.
- Does not handle the case where the strategy's z-reversion close races the bracket's stop. In live both could fire near-simultaneously; whichever fills first wins. The backtest only has the sim-exit, which approximates the strategy-close path. This is a known small modeling gap and would require live fill data to measure.

---

## If D3 flags FAIL

Two paths, depending on which way the gap goes:

1. **Time-stop adds optimism (positive gap)** — the validated edge is partly an artifact of the backtest closing losers early. Two options:
   - **(a)** Add a wall-clock time-stop watcher to `main_pairs.py` that flattens positions older than 240 min, mirroring the backtest. Cheap.
   - **(b)** Re-validate with `--sim-max-hold-minutes 999999` and accept whatever raw_p drops out. If it still survives A1 family correction, JPM/BAC stays deployed; if not, demote.

2. **Time-stop adds pessimism (negative gap)** — backtest is undercounting live edge. Less urgent but still worth re-validating against `without_ts` numbers since that's what live will actually produce.
