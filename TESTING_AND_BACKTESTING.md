# HQC Testing, Backtesting, Walk-Forward, and EOD Liquidation

## Safety-First Execution (No Live Brokerage)

The runtime now supports explicit execution-mode controls:

- `HQC_SIMULATE_ONLY=0` (default): execution router sends orders to Alpaca (paper/live depends on `TRADING_MODE`).
- `HQC_SIMULATE_ONLY=1`: execution router simulates fills and does not call Alpaca.
- `TRADING_MODE=LIVE` is ignored unless `HQC_ENABLE_LIVE_TRADING=1`.

Recommended local testing env:

```powershell
$env:TRADING_MODE = "PAPER"
$env:HQC_SIMULATE_ONLY = "1"
$env:HQC_ENABLE_LIVE_TRADING = "0"
```

Recommended Alpaca paper-trading env:

```powershell
$env:TRADING_MODE = "PAPER"
$env:HQC_SIMULATE_ONLY = "0"
$env:HQC_ENABLE_LIVE_TRADING = "0"
```

## All-Day Session Control

The live runtime now supports a built-in day-cycle scheduler:

- `HQC_SESSION_SCHEDULER=1` (default): waits outside market hours instead of entering `WARMING_UP` and timing out.
- `HQC_SESSION_PREWARM_MIN=5`: starts the runtime this many minutes before the opening bell.
- `HQC_SESSION_SHUTDOWN_DELAY_MIN=5`: stops the runtime this many minutes after the close.
- `HQC_RUN_FOREVER=1`: after a session ends, create a fresh runtime for the next session instead of exiting.

Example unattended session config:

```powershell
$env:TRADING_MODE = "PAPER"
$env:HQC_SIMULATE_ONLY = "0"
$env:HQC_SESSION_SCHEDULER = "1"
$env:HQC_RUN_FOREVER = "1"
$env:HQC_SESSION_PREWARM_MIN = "5"
$env:HQC_SESSION_SHUTDOWN_DELAY_MIN = "5"
```

## Startup Reconciliation

When live broker routing is enabled, startup now reconciles Alpaca state before trading:

- account equity updates the risk sizer
- open positions seed EOD liquidation tracking
- open positions seed virtual equity inventory state
- open Alpaca orders are restored into the order timeout/slippage monitor

This avoids starting from a false flat/no-order state after a restart.

## Watchdog Launcher

Use the watchdog to keep `main.py` running unattended and automatically restart it on failure:

```powershell
c:/HQC/.venv/Scripts/python.exe watchdog_runner.py
```

Optional watchdog controls:

- `HQC_WATCHDOG_RESTART_SEC=15`
- `HQC_WATCHDOG_MAX_RESTARTS=0` (`0` means unlimited)
- `HQC_WATCHDOG_ALWAYS_RESTART=1`

## EOD Liquidation

EOD liquidation is active via `EODLiquidationManager`:

- Tracks filled inventory from `ORDER_FILL` events.
- Auto-flattens near close (default `15:55` US/Eastern).
- Also forces flattening during graceful shutdown.

Environment controls:

```powershell
$env:HQC_EOD_LIQ_HOUR = "15"
$env:HQC_EOD_LIQ_MINUTE = "55"
```

## Offline Backtesting

Run event-driven backtest with the full signal pipeline:

```powershell
c:/HQC/.venv/Scripts/python.exe backtest_runner.py --csv state/backtest_sample.csv --symbol SPY --strategy both --output state/backtest_result.json
```

Optional benchmark + parameter overrides:

```powershell
c:/HQC/.venv/Scripts/python.exe backtest_runner.py --csv your_data.csv --symbol SPY --benchmark-symbol SPY --strategy both --min-rank-score 4.75 --orb-range-minutes 15 --vwap-tolerance-pct 0.002
```

Input CSV expected columns (aliases supported):

- timestamp (`timestamp`/`datetime`/`date`/`time`/`t`)
- symbol (`symbol`/`ticker`)
- open/high/low/close/volume (`o/h/l/c/v` aliases supported)

## Walk-Forward Evaluation

Run rolling train/test windows with rank-threshold tuning:

```powershell
c:/HQC/.venv/Scripts/python.exe walkforward_runner.py --csv state/backtest_sample.csv --symbol SPY --strategy both --train-days 20 --test-days 5 --output state/walkforward_result.json
```

The walk-forward runner now tunes a parameter grid, not just rank score. Default grids:

- `min_rank_score_grid`: `3.5,4.25,4.75,5.25`
- `orb_range_minutes_grid`: `10,15,20`
- `orb_breakout_buffer_grid`: `0.0003,0.0005,0.0008`
- `orb_min_range_pct_grid`: `0.002,0.0025,0.0035`
- `vwap_tolerance_grid`: `0.0015,0.002,0.003`
- `vwap_momentum_grid`: `0.003,0.005,0.007`

Self-improvement mode is enabled by default:

- each new walk-forward window narrows the next search around prior best parameters
- controlled via `--adaptive-keep-per-param` (default `2`)
- disable with `--no-self-improve`

Example constrained grid:

```powershell
c:/HQC/.venv/Scripts/python.exe walkforward_runner.py --csv your_data.csv --symbol SPY --benchmark-symbol SPY --strategy both --train-days 20 --test-days 5 --min-rank-score-grid 3.5,4.75 --orb-range-minutes-grid 15 --orb-breakout-buffer-grid 0.0005 --orb-min-range-pct-grid 0.0025 --vwap-tolerance-grid 0.002 --vwap-momentum-grid 0.005
```

The walk-forward output now includes:

- `selection_profit_factor`
- `candidates_evaluated`
- aggregate `avg_test_profit_factor`
- aggregate `total_candidates_evaluated`

## Performance Reporting

Generate a portfolio-style report from a backtest result:

```powershell
c:/HQC/.venv/Scripts/python.exe performance_report.py --input state/backtest_result.json --output state/performance_report.json
```

Generate a report from walk-forward output:

```powershell
c:/HQC/.venv/Scripts/python.exe performance_report.py --input state/walkforward_result.json --scope test --output state/walkforward_performance_report.json
```

The report includes:

- expectancy
- profit factor
- average trade duration
- gross exposure minutes
- total return vs benchmark return
- excess return
- annualized sharpe when enough daily observations exist
- information ratio when enough daily observations exist
- strategy-level breakdown
- top and bottom trades

## Batch Portfolio Backtesting

Run one backtest per CSV in a folder and aggregate the results:

```powershell
c:/HQC/.venv/Scripts/python.exe portfolio_batch_runner.py --input data/ --glob *.csv --strategy both --output state/portfolio_batch_result.json
```

Then generate a portfolio report:

```powershell
c:/HQC/.venv/Scripts/python.exe performance_report.py --input state/portfolio_batch_result.json --scope test --output state/portfolio_batch_report.json
```

## Notes

- Both runners use simulated execution only.
- Backtest enforces end-of-run flattening to avoid dangling overnight inventory.
- Walk-forward currently tunes `min_rank_score` grid `[3.5, 4.25, 4.75, 5.25]`.
