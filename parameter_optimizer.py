from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import itertools
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv

from core.engine.event_bus import EventBus, Event, EventType
from risk.position_sizing.confidence_scaler import DynamicRiskSizer
from strategies.orb.equity_orb import USEquityORB
from intelligence.candidate_ranker import CandidateRanker

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("EventBus").setLevel(logging.CRITICAL)
logging.getLogger("EquityORB").setLevel(logging.WARNING)
logging.getLogger("RiskPositionSizer").setLevel(logging.WARNING)
logger = logging.getLogger("ParameterOptimizer")


@dataclass
class RunResult:
    """Single backtest run result."""
    asset: str
    date: str
    range_minutes: int
    breakout_buffer_pct: float
    max_trades: int
    cooldown_bars: int
    min_range_pct: float
    base_risk_pct: float
    max_position_pct: float
    trades: int
    wins: int
    losses: int
    gross_pnl: float
    avg_pnl: float
    avg_r_multiple: float
    win_rate: float
    max_drawdown: float
    score: float


class ORBParameterOptimizer:
    """
    Monte Carlo ORB parameter optimizer.

    Features:
    - Backtests ORB strategy across grid of parameters
    - Tests on random historical trading days
    - Ranks configs by win rate, R-multiple, and PnL
    - Outputs best parameters as JSON
    - Saves detailed run logs
    
    Usage:
        optimizer = ORBParameterOptimizer(asset="SPY")
        summary = await optimizer.optimize(
            range_minutes_grid=[10, 15, 20],
            breakout_buffer_grid=[0.0003, 0.0005],
            ...
        )
    """

    def __init__(
        self,
        asset: str = "SPY",
        years_back: int = 5,
        runs_per_config: int = 5,
        seed: int = 42,
        out_dir: str = "state/optimizer",
    ) -> None:
        self.asset = asset.upper()
        self.years_back = int(years_back)
        self.runs_per_config = int(runs_per_config)
        self.random = random.Random(seed)

        self.api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing Alpaca credentials in environment.")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ORBParameterOptimizer initialized for %s (backtesting %d years)", self.asset, years_back)

    def random_trading_day(self) -> str:
        """Generate random trading day (weekday) in the past N years."""
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=365 * self.years_back)

        while True:
            offset = self.random.randint(0, (today - start_date).days)
            d = start_date + timedelta(days=offset)
            if d.weekday() < 5:  # Monday=0, Friday=4
                return d.strftime("%Y-%m-%d")

    async def fetch_historical_day(self, date_str: str) -> List[Dict[str, Any]]:
        """Fetch 1-minute bars for a single trading day from Alpaca."""
        url = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }
        params = {
            "symbols": self.asset,
            "timeframe": "1Min",
            "start": f"{date_str}T13:00:00Z",
            "end": f"{date_str}T20:30:00Z",
            "limit": 10000,
            "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
            "adjustment": "raw",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("API error for %s: %s", date_str, await resp.text())
                    return []
                data = await resp.json()
                return data.get("bars", {}).get(self.asset, [])

    def interpolate_bar_to_ticks(self, bar: Dict[str, Any]) -> List[Event]:
        """
        Expand each 1-minute bar into 4 synthetic ticks (OHLC).
        
        Allows strategy to see intrabar price movement.
        """
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
        vol = max(1.0, float(bar["v"]) / 4.0)

        dt = datetime.strptime(bar["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        base_ts = int(dt.timestamp() * 1000)

        # Order ticks based on close vs open
        prices = [o, l, h, c] if c > o else [o, h, l, c]
        ticks: List[Event] = []
        for i, px in enumerate(prices):
            ticks.append(
                Event(
                    type=EventType.TICK,
                    payload={
                        "ticker": self.asset,
                        "symbol": self.asset,
                        "price": px,
                        "volume": vol,
                        "timestamp": base_ts + (i * 15000),
                    },
                )
            )
        return ticks

    async def run_single_day(
        self,
        date_str: str,
        *,
        range_minutes: int,
        breakout_buffer_pct: float,
        max_trades: int,
        cooldown_bars: int,
        min_range_pct: float,
        base_risk_pct: float,
        max_position_pct: float,
    ) -> Optional[RunResult]:
        """
        Run backtest for one day with given parameters.
        
        Returns:
            RunResult with performance metrics
        """
        bars = await self.fetch_historical_day(date_str)
        if not bars:
            return None

        bus = EventBus()
        await bus.start()

        strategy = USEquityORB(
            target_asset=self.asset,
            bus=bus,
            range_minutes=range_minutes,
            max_trades=max_trades,
            cooldown_bars=cooldown_bars,
            min_range_pct=min_range_pct,
            breakout_buffer_pct=breakout_buffer_pct,
        )
        _ = strategy

        ranker = CandidateRanker(
            bus=bus,
            benchmark=self.asset,
            min_score=0.0,
            decisions_path=str(self.out_dir / "backtest_decisions.jsonl")
        )
        _ = ranker

        sizer = DynamicRiskSizer(
            bus=bus,
            account_equity=100000.0,
            base_risk_pct=base_risk_pct,
            max_position_pct=max_position_pct,
        )
        _ = sizer

        raw_orders: List[Dict[str, Any]] = []
        sized_orders: List[Dict[str, Any]] = []

        async def order_catcher(event: Event) -> None:
            payload = dict(event.payload or {})

            is_order_like = (
                payload.get("asset") == self.asset
                and payload.get("action") is not None
                and (
                    payload.get("reference_price") is not None
                    or payload.get("entry_price") is not None
                )
            )
            if not is_order_like:
                return

            if "shares" in payload and int(payload.get("shares", 0) or 0) > 0:
                sized_orders.append(payload)
            else:
                raw_orders.append(payload)

        bus.subscribe(EventType.ORDER_CREATE, order_catcher)

        all_ticks: List[Event] = []
        for bar in bars:
            all_ticks.extend(self.interpolate_bar_to_ticks(bar))

        for tick in all_ticks:
            bus.publish(tick)
            await asyncio.sleep(0)

        await bus._queue.join()
        await asyncio.sleep(0.05)

        logger.debug(
            "[DEBUG] date=%s raw_orders=%d sized_orders=%d",
            date_str,
            len(raw_orders),
            len(sized_orders),
        )

        if sized_orders:
            logger.debug("[SCORING] Using %d SIZED orders from DynamicRiskSizer.", len(sized_orders))
            orders_for_scoring = sized_orders
        elif raw_orders:
            logger.warning(
                "[FALLBACK] DynamicRiskSizer produced 0 sized orders. "
                "Falling back to manual promotion of %d raw orders.",
                len(raw_orders),
            )
            orders_for_scoring = self._promote_raw_orders(
                raw_orders=raw_orders,
                account_equity=100000.0,
                base_risk_pct=base_risk_pct,
                max_position_pct=max_position_pct,
            )
        else:
            logger.debug("[NO ORDERS] Date %s produced 0 signals.", date_str)
            orders_for_scoring = []

        scored_trades = self._score_orders_against_bars(orders_for_scoring, bars)

        await bus.stop()

        if not scored_trades:
            return RunResult(
                asset=self.asset,
                date=date_str,
                range_minutes=range_minutes,
                breakout_buffer_pct=breakout_buffer_pct,
                max_trades=max_trades,
                cooldown_bars=cooldown_bars,
                min_range_pct=min_range_pct,
                base_risk_pct=base_risk_pct,
                max_position_pct=max_position_pct,
                trades=0,
                wins=0,
                losses=0,
                gross_pnl=0.0,
                avg_pnl=0.0,
                avg_r_multiple=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                score=-0.50,
            )

        gross_pnl = sum(t["pnl"] for t in scored_trades)
        wins = sum(1 for t in scored_trades if t["pnl"] > 0)
        losses = sum(1 for t in scored_trades if t["pnl"] < 0)
        trades = len(scored_trades)
        avg_pnl = gross_pnl / trades if trades else 0.0
        avg_r = sum(t["r_multiple"] for t in scored_trades) / trades if trades else 0.0
        win_rate = wins / trades if trades else 0.0
        max_dd = self._equity_curve_max_drawdown([t["pnl"] for t in scored_trades])

        trade_participation_bonus = min(0.60, trades * 0.08)
        score = (
            (avg_r * 2.75)
            + (win_rate * 1.25)
            + (gross_pnl / 4000.0)
            + trade_participation_bonus
            - (abs(max_dd) / 3000.0)
        )

        return RunResult(
            asset=self.asset,
            date=date_str,
            range_minutes=range_minutes,
            breakout_buffer_pct=breakout_buffer_pct,
            max_trades=max_trades,
            cooldown_bars=cooldown_bars,
            min_range_pct=min_range_pct,
            base_risk_pct=base_risk_pct,
            max_position_pct=max_position_pct,
            trades=trades,
            wins=wins,
            losses=losses,
            gross_pnl=round(gross_pnl, 2),
            avg_pnl=round(avg_pnl, 2),
            avg_r_multiple=round(avg_r, 4),
            win_rate=round(win_rate, 4),
            max_drawdown=round(max_dd, 2),
            score=round(score, 4),
        )

    def _promote_raw_orders(
        self,
        raw_orders: List[Dict[str, Any]],
        *,
        account_equity: float,
        base_risk_pct: float,
        max_position_pct: float,
    ) -> List[Dict[str, Any]]:
        """
        Manually calculate position sizes for raw strategy signals.
        
        Used as fallback if the risk sizer didn't produce sized orders.
        """
        promoted: List[Dict[str, Any]] = []

        for payload in raw_orders:
            entry_price = self._to_float(payload.get("entry_price", payload.get("reference_price")), 0.0)
            stop_loss_price = self._to_float(payload.get("stop_loss_price", payload.get("stop_loss")), 0.0)

            if entry_price <= 0 or stop_loss_price <= 0:
                continue

            per_share_risk = abs(entry_price - stop_loss_price)
            if per_share_risk <= 0:
                continue

            dollar_risk = account_equity * base_risk_pct
            raw_shares = dollar_risk / per_share_risk

            max_position_dollars = account_equity * max_position_pct
            max_position_shares = max(1, int(max_position_dollars // entry_price))
            final_shares = min(int(raw_shares), max_position_shares)

            if final_shares <= 0:
                continue

            promoted.append(
                {
                    **payload,
                    "stage": "SIZED_FALLBACK",
                    "shares": final_shares,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "capital_allocated": round(final_shares * entry_price, 2),
                    "risk_dollars": round(final_shares * per_share_risk, 2),
                }
            )

        return promoted

    def _score_orders_against_bars(
        self,
        orders: List[Dict[str, Any]],
        bars: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """
        Score each order: did it hit target or stop loss?
        
        Assumes 2:1 risk:reward ratio.
        """
        if not orders:
            return []

        bars_df = pd.DataFrame(bars).copy()
        bars_df["ts"] = pd.to_datetime(bars_df["t"], utc=True)
        bars_df = bars_df.sort_values("ts").reset_index(drop=True)

        scored: List[Dict[str, float]] = []

        for order in orders:
            entry = float(order["entry_price"])
            stop = float(order.get("stop_loss_price", order.get("stop_loss", 0.0)))
            shares = int(order["shares"])
            action = str(order["action"]).upper()
            entry_ts_ms = self._extract_entry_ts_ms(order, bars_df)

            if stop <= 0 or shares <= 0 or entry <= 0:
                continue

            risk_per_share = abs(entry - stop)
            if risk_per_share <= 0:
                continue

            target = entry + (2.0 * risk_per_share if action == "BUY" else -2.0 * risk_per_share)
            future_bars = bars_df[(bars_df["ts"].astype("int64") // 10**6) >= entry_ts_ms]

            pnl = 0.0
            r_multiple = 0.0
            resolved = False

            for _, row in future_bars.iterrows():
                hi = float(row["h"])
                lo = float(row["l"])

                if action == "BUY":
                    hit_stop = lo <= stop
                    hit_target = hi >= target
                    if hit_stop and hit_target:
                        pnl = (stop - entry) * shares
                        r_multiple = -1.0
                        resolved = True
                        break
                    if hit_stop:
                        pnl = (stop - entry) * shares
                        r_multiple = -1.0
                        resolved = True
                        break
                    if hit_target:
                        pnl = (target - entry) * shares
                        r_multiple = 2.0
                        resolved = True
                        break
                else:
                    hit_stop = hi >= stop
                    hit_target = lo <= target
                    if hit_stop and hit_target:
                        pnl = (entry - stop) * shares
                        r_multiple = -1.0
                        resolved = True
                        break
                    if hit_stop:
                        pnl = (entry - stop) * shares
                        r_multiple = -1.0
                        resolved = True
                        break
                    if hit_target:
                        pnl = (entry - target) * shares
                        r_multiple = 2.0
                        resolved = True
                        break

            if not resolved and not future_bars.empty:
                last_close = float(future_bars.iloc[-1]["c"])
                pnl = (last_close - entry) * shares if action == "BUY" else (entry - last_close) * shares
                r_multiple = pnl / (risk_per_share * shares)

            scored.append({"pnl": float(pnl), "r_multiple": float(r_multiple)})

        return scored

    @staticmethod
    def _extract_entry_ts_ms(order: Dict[str, Any], bars_df: pd.DataFrame) -> int:
        """Extract order entry timestamp, with fallback to first bar."""
        ts = None
        
        if order.get("timestamp") is not None:
            try:
                ts = int(order["timestamp"])
            except Exception:
                pass

        if ts is None:
            ctx = order.get("signal_context", {}) or {}
            if ctx.get("timestamp") is not None:
                try:
                    ts = int(ctx["timestamp"])
                except Exception:
                    pass

        max_valid_ts = int(bars_df.iloc[-1]["ts"].value // 10**6)
        
        if ts is None or ts > max_valid_ts:
            return int(bars_df.iloc[0]["ts"].value // 10**6)

        return ts

    @staticmethod
    def _equity_curve_max_drawdown(pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            dd = equity - peak
            max_dd = min(max_dd, dd)
        return max_dd

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """Safe float conversion."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def optimize(
        self,
        *,
        range_minutes_grid: List[int],
        breakout_buffer_grid: List[float],
        max_trades_grid: List[int],
        cooldown_bars_grid: List[int],
        min_range_pct_grid: List[float],
        base_risk_pct_grid: List[float],
        max_position_pct_grid: List[float],
    ) -> pd.DataFrame:
        """
        Run full parameter optimization grid search.
        
        Returns:
            Summary DataFrame with top configs ranked by score
        """
        configs = list(
            itertools.product(
                range_minutes_grid,
                breakout_buffer_grid,
                max_trades_grid,
                cooldown_bars_grid,
                min_range_pct_grid,
                base_risk_pct_grid,
                max_position_pct_grid,
            )
        )

        logger.info("Running %d configs x %d random days", len(configs), self.runs_per_config)
        all_results: List[RunResult] = []

        for idx, config in enumerate(configs, start=1):
            (
                range_minutes,
                breakout_buffer_pct,
                max_trades,
                cooldown_bars,
                min_range_pct,
                base_risk_pct,
                max_position_pct,
            ) = config

            logger.info(
                "[%d/%d] Testing config: range=%s breakout_buf=%.4f max_trades=%s cooldown=%s min_range=%.4f risk=%.4f max_pos=%.2f",
                idx,
                len(configs),
                range_minutes,
                breakout_buffer_pct,
                max_trades,
                cooldown_bars,
                min_range_pct,
                base_risk_pct,
                max_position_pct,
            )

            for _ in range(self.runs_per_config):
                date_str = self.random_trading_day()
                result = await self.run_single_day(
                    date_str,
                    range_minutes=range_minutes,
                    breakout_buffer_pct=breakout_buffer_pct,
                    max_trades=max_trades,
                    cooldown_bars=cooldown_bars,
                    min_range_pct=min_range_pct,
                    base_risk_pct=base_risk_pct,
                    max_position_pct=max_position_pct,
                )
                if result is not None:
                    all_results.append(result)

        df = pd.DataFrame([asdict(r) for r in all_results])
        if df.empty:
            raise RuntimeError("No optimizer results were produced.")

        summary = (
            df.groupby(
                [
                    "range_minutes",
                    "breakout_buffer_pct",
                    "max_trades",
                    "cooldown_bars",
                    "min_range_pct",
                    "base_risk_pct",
                    "max_position_pct",
                ],
                as_index=False,
            )
            .agg(
                runs=("date", "count"),
                trades=("trades", "mean"),
                wins=("wins", "mean"),
                losses=("losses", "mean"),
                gross_pnl=("gross_pnl", "mean"),
                avg_pnl=("avg_pnl", "mean"),
                avg_r_multiple=("avg_r_multiple", "mean"),
                win_rate=("win_rate", "mean"),
                max_drawdown=("max_drawdown", "mean"),
                score=("score", "mean"),
            )
            .sort_values(
                ["score", "avg_r_multiple", "gross_pnl", "win_rate", "trades"],
                ascending=[False, False, False, False, False],
            )
            .reset_index(drop=True)
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_path = self.out_dir / f"orb_optimizer_runs_{ts}.csv"
        summary_path = self.out_dir / f"orb_optimizer_summary_{ts}.csv"
        best_json_path = self.out_dir / f"best_orb_params_{ts}.json"

        df.to_csv(runs_path, index=False)
        summary.to_csv(summary_path, index=False)

        best_row = summary.iloc[0].to_dict()
        best_payload = {
            "asset": self.asset,
            "generated_at": ts,
            "runs_per_config": self.runs_per_config,
            "recommended_params": {
                "range_minutes": int(best_row["range_minutes"]),
                "breakout_buffer_pct": float(best_row["breakout_buffer_pct"]),
                "max_trades": int(best_row["max_trades"]),
                "cooldown_bars": int(best_row["cooldown_bars"]),
                "min_range_pct": float(best_row["min_range_pct"]),
                "base_risk_pct": float(best_row["base_risk_pct"]),
                "max_position_pct": float(best_row["max_position_pct"]),
            },
            "summary_metrics": {
                "runs": int(best_row["runs"]),
                "trades": float(best_row["trades"]),
                "avg_r_multiple": float(best_row["avg_r_multiple"]),
                "win_rate": float(best_row["win_rate"]),
                "gross_pnl": float(best_row["gross_pnl"]),
                "max_drawdown": float(best_row["max_drawdown"]),
                "score": float(best_row["score"]),
            },
        }

        best_json_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

        logger.info("Saved run-level results: %s", runs_path)
        logger.info("Saved summary results: %s", summary_path)
        logger.info("Saved best params JSON: %s", best_json_path)

        return summary


async def main() -> None:
    """Run parameter optimization from command line."""
    optimizer = ORBParameterOptimizer(
        asset=os.getenv("HQC_OPT_ASSET", "SPY"),
        years_back=int(os.getenv("HQC_OPT_YEARS_BACK", "5")),
        runs_per_config=int(os.getenv("HQC_OPT_RUNS_PER_CONFIG", "5")),
        seed=int(os.getenv("HQC_OPT_SEED", "42")),
    )

    summary = await optimizer.optimize(
        # --- FIXED: Much wider, more realistic grid ---
        range_minutes_grid=[5, 10, 15, 20],           # Was [10, 15]
        breakout_buffer_grid=[0.001, 0.003, 0.005],  # Was [0.0003, 0.0005]
        max_trades_grid=[1, 2, 3],                     # Was [1]
        cooldown_bars_grid=[3, 5, 8],                  # Was [5, 10]
        min_range_pct_grid=[0.001, 0.002, 0.003],     # Was [0.0015, 0.0020, 0.0025]
        base_risk_pct_grid=[0.005, 0.01, 0.015],      # Was [0.0075, 0.01]
        max_position_pct_grid=[0.10, 0.15, 0.20],     # Was [0.15, 0.20]
    )

    print("\nTop 15 configs:\n")
    print(summary.head(15).to_string(index=False))

    print("\nSuggested next step:")
    print("Take the top 5 configs and rerun them with HQC_OPT_RUNS_PER_CONFIG=25 for stronger confidence.")


if __name__ == "__main__":
    asyncio.run(main())