from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from core.engine.event_bus import Event, EventBus, EventType
from core.engine.session_clock import MarketSessionClock, SessionPhase
from core.engine.state_machine import GlobalStateMachine, SystemState
from core.execution.broker_router import AlpacaExecutionRouter
from core.execution.eod_liquidator import EODLiquidationManager
from core.execution.router_fallback import ResilientFallbackManager
from core.execution.slippage_controller import SlippageController
from core.feedback.unified_logger import UnifiedFeedbackLogger
from data.feeds.ws_manager import AlpacaWebsocketManager
from intelligence.candidate_ranker import CandidateRanker
from intelligence.regime_detection.hmm_classifier import RegimeClassifierHMM
from intelligence.ml_pipeline.model_retraining import ContinuousRetrainingPipeline
from risk.position_sizing.confidence_scaler import DynamicRiskSizer
from risk.virtual_monitor.equity_slope_detector import VirtualEquitySlopeDetector
from strategies.orb.equity_orb import USEquityORB
from strategies.vwap.hunter_state_machine import USEquityVWAPHunter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SystemMain")


class TradingNode:
    """
    HQC bootstrap and supervision node with learning modules.

    Includes:
    - Alpaca market data feed
    - Alpaca execution
    - Candidate ranking + dynamic sizing
    - HMM regime classification
    - Continuous ML retraining
    - Feed staleness / warming-up protection
    """

    def __init__(self) -> None:
        self._last_tick_timestamp_ms: int = int(time.time() * 1000)
        self._max_tick_staleness_sec: float = float(os.getenv("HQC_MAX_TICK_STALENESS_SEC", "60"))
        self._stale_check_task: Optional[asyncio.Task] = None

        self._warming_up_start_time: Optional[float] = None
        self._warming_up_timeout_sec: float = 300.0
        self._warming_up_check_task: Optional[asyncio.Task] = None
        self._session_monitor_task: Optional[asyncio.Task] = None
        self._last_session_wait_log: Optional[str] = None
        self._calendar_last_refresh: Optional[datetime] = None

        self._feed_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._ml_retrain_task: Optional[asyncio.Task] = None

        self._shutdown_started = False
        self._seen_first_tick = False
        self._last_tick_symbol: Optional[str] = None
        self._last_tick_ts: Optional[str] = None
        self._ml_confidence: Dict[str, float] = {}

        self.bus = EventBus()

        universe_env = os.getenv("HQC_UNIVERSE", "SPY,QQQ,TSLA")
        self.target_symbols: List[str] = [s.strip().upper() for s in universe_env.split(",") if s.strip()]
        if not self.target_symbols:
            self.target_symbols = ["SPY", "QQQ", "TSLA"]

        self.initial_capital = float(os.getenv("HQC_INITIAL_CAPITAL", "100000"))

        trading_mode_raw = os.getenv("TRADING_MODE", "PAPER").upper().strip()
        valid_modes = {"PAPER", "LIVE"}
        if trading_mode_raw not in valid_modes:
            raise ValueError(
                f"Invalid TRADING_MODE={trading_mode_raw}. "
                f"Must be one of {valid_modes}. Check your .env file."
            )
        if trading_mode_raw == "LIVE" and os.getenv("HQC_ENABLE_LIVE_TRADING", "0") != "1":
            logger.warning("LIVE mode requested but HQC_ENABLE_LIVE_TRADING!=1. Falling back to PAPER for safety.")
            trading_mode_raw = "PAPER"
        self.is_paper = trading_mode_raw == "PAPER"
        self.simulate_only = self._env_flag("HQC_SIMULATE_ONLY", default=False)
        self.run_forever = self._env_flag("HQC_RUN_FOREVER", default=False)
        self.session_scheduler_enabled = self._env_flag("HQC_SESSION_SCHEDULER", default=True)
        logger.info("TRADING_MODE validated: %s", trading_mode_raw)

        self.api_key = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID", "YOUR_PAPER_KEY"))
        self.api_secret = os.getenv("ALPACA_API_SECRET", os.getenv("APCA_API_SECRET_KEY", "YOUR_PAPER_SECRET"))
        self.market_data_feed = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
        if self.market_data_feed not in {"iex", "sip"}:
            raise ValueError("ALPACA_DATA_FEED must be 'iex' or 'sip'.")

        if self._looks_like_placeholder(self.api_key) or self._looks_like_placeholder(self.api_secret):
            raise ValueError(
                "Alpaca credentials are required for both execution and market data. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET (or APCA_API_KEY_ID/APCA_API_SECRET_KEY)."
            )

        if self.simulate_only:
            logger.warning("HQC_SIMULATE_ONLY=1. Orders will be simulated and will not be sent to Alpaca.")

        self.session_clock = MarketSessionClock(
            timezone_name=os.getenv("HQC_TIMEZONE", "US/Eastern"),
            market_open_hour=int(os.getenv("HQC_MARKET_OPEN_HOUR", "9")),
            market_open_minute=int(os.getenv("HQC_MARKET_OPEN_MINUTE", "30")),
            market_close_hour=int(os.getenv("HQC_MARKET_CLOSE_HOUR", "16")),
            market_close_minute=int(os.getenv("HQC_MARKET_CLOSE_MINUTE", "0")),
            prewarm_minutes=int(os.getenv("HQC_SESSION_PREWARM_MIN", "5")),
            shutdown_delay_minutes=int(os.getenv("HQC_SESSION_SHUTDOWN_DELAY_MIN", "5")),
        )

        Path("state").mkdir(parents=True, exist_ok=True)
        Path("state/feedback").mkdir(parents=True, exist_ok=True)

        self.feedback = UnifiedFeedbackLogger(
            root="state/feedback",
            system_name="HQC",
            arm="equities",
            env="paper" if self.is_paper else "live",
        )

        self.state_machine = GlobalStateMachine(
            self.bus,
            system_name="HQC",
            arm="equities",
            env="paper" if self.is_paper else "live",
        )

        self.ranker = CandidateRanker(
            bus=self.bus,
            benchmark=self.target_symbols[0],
            min_score=float(os.getenv("HQC_MIN_SCORE", "4.75")),
            max_spread_bps=float(os.getenv("HQC_MAX_SPREAD_BPS", "18")),
            max_dist_vwap_pct=float(os.getenv("HQC_MAX_DIST_VWAP", "0.012")),
            decisions_path="state/feedback/decisions.jsonl",
        )

        self.risk_sizer = DynamicRiskSizer(
            bus=self.bus,
            account_equity=self.initial_capital,
            base_risk_pct=float(os.getenv("HQC_RISK_PCT", "0.01")),
            max_position_pct=float(os.getenv("HQC_MAX_POSITION_PCT", "0.20")),
        )

        self.slippage_controller = SlippageController(
            bus=self.bus,
            max_slippage_bps=float(os.getenv("HQC_MAX_SLIPPAGE_BPS", "8")),
            max_hanging_time_sec=float(os.getenv("HQC_MAX_HANGING_SEC", "10")),
        )

        self.resilience_manager = ResilientFallbackManager(
            bus=self.bus,
            max_retries=3,
            base_backoff_sec=0.2,
            system_name="HQC",
            arm="equities",
            env="paper" if self.is_paper else "live",
        )

        self.execution_router = AlpacaExecutionRouter(
            api_key=self.api_key,
            api_secret=self.api_secret,
            bus=self.bus,
            is_paper=self.is_paper,
            simulate_only=self.simulate_only,
            resilient_manager=self.resilience_manager,
            slippage_controller=self.slippage_controller,
        )

        self.regime_classifier = RegimeClassifierHMM(n_components=3)
        self._regime_classifier_fitted = False
        logger.info("Regime Classifier HMM initialized (3-component)")

        self.virtual_equity_monitor = VirtualEquitySlopeDetector(
            bus=self.bus,
            initial_capital=self.initial_capital,
            slope_lookback=20,
            system_name="HQC",
            arm="equities",
            env="paper" if self.is_paper else "live",
        )

        self.eod_liquidator = EODLiquidationManager(
            bus=self.bus,
            liquidate_hour=int(os.getenv("HQC_EOD_LIQ_HOUR", "15")),
            liquidate_minute=int(os.getenv("HQC_EOD_LIQ_MINUTE", "55")),
        )

        self.ml_retrainer: Dict[str, ContinuousRetrainingPipeline] = {}
        for symbol in self.target_symbols:
            self.ml_retrainer[symbol] = ContinuousRetrainingPipeline(
                bus=self.bus,
                target_asset=symbol,
                retrain_interval_sec=float(os.getenv("HQC_ML_RETRAIN_SEC", "3600.0")),
                min_training_rows=500,
            )
        logger.info("ML Retraining pipelines initialized for %s", self.target_symbols)

        from collections import deque as _deque
        self._recent_ohlcv: Dict[str, Dict[str, _deque]] = {}
        for symbol in self.target_symbols:
            self._recent_ohlcv[symbol] = {
                "open": _deque(maxlen=300),
                "high": _deque(maxlen=300),
                "low": _deque(maxlen=300),
                "close": _deque(maxlen=300),
                "volume": _deque(maxlen=300),
            }

        self.strategies = []
        for symbol in self.target_symbols:
            self.strategies.append(
                USEquityORB(
                    target_asset=symbol,
                    bus=self.bus,
                    range_minutes=15,
                    max_trades=2,
                    min_range_pct=float(os.getenv("HQC_ORB_MIN_RANGE_PCT", "0.0025")),
                    breakout_buffer_pct=float(os.getenv("HQC_ORB_BREAKOUT_BUFFER_PCT", "0.0005")),
                    breakout_confirmation_ticks=int(os.getenv("HQC_ORB_CONFIRMATION_TICKS", "2")),
                    fakeout_reset_pct=float(os.getenv("HQC_ORB_FAKEOUT_RESET_PCT", "0.0003")),
                    enable_shorts=self._env_flag("HQC_ENABLE_SHORTS", default=True),
                )
            )
            self.strategies.append(
                USEquityVWAPHunter(
                    target_asset=symbol,
                    bus=self.bus,
                    min_volume_shares=float(os.getenv("HQC_VWAP_MIN_VOLUME", "250000")),
                    vwap_tolerance_pct=float(os.getenv("HQC_VWAP_TOLERANCE_PCT", "0.002")),
                    momentum_threshold_pct=float(os.getenv("HQC_VWAP_MOMENTUM_PCT", "0.005")),
                    max_daily_trades=int(os.getenv("HQC_VWAP_MAX_DAILY_TRADES", "3")),
                    cooldown_bars=int(os.getenv("HQC_VWAP_COOLDOWN_BARS", "8")),
                    min_stop_pct=float(os.getenv("HQC_VWAP_MIN_STOP_PCT", "0.003")),
                    max_window_bars=int(os.getenv("HQC_VWAP_MAX_WINDOW_BARS", "8")),
                    bounce_confirmation_ticks=int(os.getenv("HQC_VWAP_CONFIRMATION_TICKS", "2")),
                )
            )

        self.data_feed = AlpacaWebsocketManager(
            api_key=self.api_key,
            api_secret=self.api_secret,
            symbols=sorted(set(self.target_symbols + [self.target_symbols[0]])),
            bus=self.bus,
            feed=self.market_data_feed,
        )

        self.bus.subscribe(EventType.TICK, self.on_tick)
        self.bus.subscribe(EventType.TICK, self.on_first_tick)
        self.bus.subscribe(EventType.MODEL_UPDATE, self.on_model_update)

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _looks_like_placeholder(value: Optional[str]) -> bool:
        text = str(value or "").strip()
        if not text:
            return True
        upper = text.upper()
        placeholders = {
            "YOUR_PAPER_KEY",
            "YOUR_PAPER_SECRET",
            "YOUR_API_KEY",
            "YOUR_API_SECRET",
            "CHANGE_ME",
        }
        return (
            upper in placeholders
            or "YOUR_" in upper
            or "PLACEHOLDER" in upper
            or upper.startswith("PK_YOUR")
        )

    async def _wait_for_session_window(self) -> None:
        if not self.session_scheduler_enabled:
            return

        await self._refresh_market_calendar(force=True)

        while not self._shutdown_started:
            now_est = self.session_clock.now()
            if self._calendar_last_refresh is None or (now_est - self._calendar_last_refresh).total_seconds() >= 3600:
                await self._refresh_market_calendar()

            phase = self.session_clock.phase(now_est)
            if phase in {SessionPhase.PREWARM, SessionPhase.IN_SESSION}:
                self._last_session_wait_log = None
                return

            next_start = self.session_clock.next_prewarm_start(now_est)
            message = (
                f"Outside market session ({phase.value}). Waiting until "
                f"{next_start.strftime('%Y-%m-%d %I:%M:%S %p %Z')}"
            )
            if self._last_session_wait_log != message:
                logger.info(message)
                self._last_session_wait_log = message

            sleep_sec = min(max(1.0, (next_start - now_est).total_seconds()), 300.0)
            await asyncio.sleep(sleep_sec)

    async def _session_window_monitor(self) -> None:
        if not self.session_scheduler_enabled:
            return

        while not self._shutdown_started and self.state_machine.current_state != SystemState.HALTED:
            now_est = self.session_clock.now()
            if self._calendar_last_refresh is None or (now_est - self._calendar_last_refresh).total_seconds() >= 3600:
                await self._refresh_market_calendar()

            phase = self.session_clock.phase(now_est)
            if phase == SessionPhase.AFTER_HOURS:
                logger.warning("Session window ended at %s. Halting live runtime.", now_est.strftime("%I:%M:%S %p %Z"))
                self.state_machine.transition_to(
                    SystemState.HALTED,
                    "Market session ended. Shutting down until next session.",
                )
                break
            await asyncio.sleep(30.0)

    async def _refresh_market_calendar(self, force: bool = False) -> None:
        if not self.session_scheduler_enabled:
            return

        now_est = self.session_clock.now()
        if not force and self._calendar_last_refresh is not None:
            if (now_est - self._calendar_last_refresh).total_seconds() < 3600:
                return

        subdomain = "paper-api" if self.is_paper else "api"
        endpoint = f"https://{subdomain}.alpaca.markets/v2/calendar"

        start_day = now_est.date()
        days_ahead = int(os.getenv("HQC_CALENDAR_LOOKAHEAD_DAYS", "14"))
        end_day = start_day + timedelta(days=days_ahead)
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }
        params = {
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
        }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        logger.warning("Calendar refresh failed: status=%s body=%s", resp.status, body[:300])
                        return
                    try:
                        entries = await resp.json(content_type=None)
                    except Exception:
                        logger.warning("Calendar refresh parse failure: %s", body[:300])
                        return
        except Exception as exc:
            logger.warning("Calendar refresh error: %s", exc)
            return

        if not isinstance(entries, list):
            return

        open_days: set[date] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                trading_date = date.fromisoformat(str(entry.get("date")))
                open_text = str(entry.get("open", "09:30"))
                close_text = str(entry.get("close", "16:00"))
                open_hour, open_minute = (int(x) for x in open_text.split(":"))
                close_hour, close_minute = (int(x) for x in close_text.split(":"))
            except Exception:
                continue

            self.session_clock.set_trading_day_override(
                trading_date=trading_date,
                market_open_hour=open_hour,
                market_open_minute=open_minute,
                market_close_hour=close_hour,
                market_close_minute=close_minute,
            )
            open_days.add(trading_date)

        day_cursor = start_day
        while day_cursor <= end_day:
            if day_cursor.weekday() < 5 and day_cursor not in open_days:
                self.session_clock.set_market_closed(day_cursor)
            day_cursor += timedelta(days=1)

        self._calendar_last_refresh = now_est
        logger.info("Market calendar refreshed: %d open days loaded (%s to %s)", len(open_days), start_day, end_day)

    async def _reconcile_broker_state(self) -> None:
        if self.simulate_only:
            return

        snapshot = await self.execution_router.reconcile_startup_state()
        account_equity = float(snapshot.get("account_equity", 0.0) or 0.0)
        if account_equity > 0:
            self.risk_sizer.update_equity(account_equity)

        positions = snapshot.get("positions") or {}
        self.eod_liquidator.seed_positions(positions)
        self.virtual_equity_monitor.seed_positions(positions, current_equity=account_equity or None)

        logger.info(
            "Startup reconciliation complete: equity=$%.2f positions=%d open_orders=%d",
            account_equity,
            len(positions),
            len(snapshot.get("open_orders") or []),
        )

    async def _check_warming_up_timeout(self) -> None:
        while self.state_machine.current_state != SystemState.HALTED and not self._shutdown_started:
            await asyncio.sleep(10.0)

            if self.state_machine.current_state == SystemState.WARMING_UP:
                if self._warming_up_start_time is None:
                    self._warming_up_start_time = asyncio.get_event_loop().time()

                elapsed_sec = asyncio.get_event_loop().time() - self._warming_up_start_time
                if elapsed_sec > self._warming_up_timeout_sec:
                    logger.error(
                        "System stuck in WARMING_UP for %.1f seconds. No market data received. Auto-halting.",
                        elapsed_sec,
                    )
                    self.state_machine.transition_to(
                        SystemState.HALTED,
                        f"WARMING_UP timeout after {elapsed_sec:.1f}s (no first tick)",
                    )
                    break
            else:
                self._warming_up_start_time = None

    async def _check_feed_staleness(self) -> None:
        while self.state_machine.current_state != SystemState.HALTED and not self._shutdown_started:
            await asyncio.sleep(5.0)

            current_time_ms = int(time.time() * 1000)
            staleness_ms = current_time_ms - self._last_tick_timestamp_ms
            staleness_sec = staleness_ms / 1000.0

            if staleness_sec > self._max_tick_staleness_sec:
                logger.error(
                    "Data feed is STALE. No tick received for %.1f seconds. Last tick symbol: %s at %s",
                    staleness_sec,
                    self._last_tick_symbol,
                    self._last_tick_ts,
                )
                self.state_machine.transition_to(
                    SystemState.HALTED,
                    f"Data feed stale for {staleness_sec:.1f}s (threshold: {self._max_tick_staleness_sec}s)",
                )
                break

    async def on_tick(self, event: Event) -> None:
        try:
            payload = event.payload or {}
            symbol = payload.get("ticker") or payload.get("symbol")
            if symbol:
                symbol = str(symbol).upper()

            # Feed heartbeat must track local receipt time, not exchange event time.
            # Some brokers may emit delayed trade timestamps that would otherwise
            # trigger false stale-feed halts.
            self._last_tick_timestamp_ms = int(time.time() * 1000)

            self._last_tick_symbol = symbol
            ts_val = payload.get("timestamp")
            self._last_tick_ts = str(ts_val) if ts_val is not None else None

            try:
                price = float(payload.get("price", 0))
                volume = float(payload.get("volume", 0))

                if symbol and symbol in self._recent_ohlcv and price > 0:
                    self._recent_ohlcv[symbol]["open"].append(price)
                    self._recent_ohlcv[symbol]["high"].append(price)
                    self._recent_ohlcv[symbol]["low"].append(price)
                    self._recent_ohlcv[symbol]["close"].append(price)
                    self._recent_ohlcv[symbol]["volume"].append(volume)
            except Exception as exc:
                logger.debug("Failed to accumulate OHLCV: %s", exc)

        except Exception as exc:
            logger.error("Unhandled exception in on_tick: %s", exc, exc_info=True)

    async def on_first_tick(self, event: Event) -> None:
        if self._seen_first_tick:
            return

        try:
            payload = event.payload or {}
            symbol = payload.get("ticker") or payload.get("symbol")
            price = payload.get("price")

            self._seen_first_tick = True
            logger.info(
                "First market-data tick received: symbol=%s price=%s. Promoting system to LIVE_TRADING.",
                symbol,
                price,
            )

            if self.state_machine.current_state == SystemState.WARMING_UP:
                self.state_machine.transition_to(
                    SystemState.LIVE_TRADING,
                    f"First tick received for {symbol or 'unknown symbol'}.",
                )
                asyncio.create_task(self._fit_regime_classifier_async(), name="fit_regime_classifier")
        except Exception as exc:
            logger.error("Unhandled exception in on_first_tick: %s", exc, exc_info=True)

    async def _fit_regime_classifier_async(self) -> None:
        try:
            logger.info("Starting HMM regime classifier fitting...")

            historical_df = await self._fetch_historical_data_for_fitting()
            if historical_df is None or len(historical_df) < 50:
                logger.warning("Insufficient historical data to fit HMM classifier")
                return

            await asyncio.to_thread(self.regime_classifier.fit_predict, historical_df)
            self._regime_classifier_fitted = True
            logger.info("HMM regime classifier fitted successfully")

        except Exception as exc:
            logger.error("Failed to fit regime classifier: %s", exc, exc_info=True)

    async def _fetch_historical_data_for_fitting(self):
        import pandas as pd
        import aiohttp
        from datetime import datetime, timedelta

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=100)

        url = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }
        params = {
            "symbols": self.target_symbols[0],
            "timeframe": "1Day",
            "start": f"{start_date.isoformat()}T00:00:00Z",
            "end": f"{end_date.isoformat()}T23:59:59Z",
            "limit": 1000,
            "feed": self.market_data_feed,
            "adjustment": "raw",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    raw_text = await resp.text()

                    if resp.status != 200:
                        logger.warning(
                            "Failed to fetch historical data for %s: status=%s body=%s",
                            self.target_symbols[0],
                            resp.status,
                            raw_text[:500],
                        )
                        return None

                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        logger.error(
                            "Failed to parse historical JSON. Content-Type=%s Body=%s",
                            resp.headers.get("Content-Type"),
                            raw_text[:500],
                        )
                        return None

            history = ((data or {}).get("bars") or {}).get(self.target_symbols[0], [])
            if not history:
                return None

            if isinstance(history, dict):
                history = [history]

            df = pd.DataFrame(history)
            if df.empty:
                return None

            df = df.rename(columns={"c": "close", "h": "high", "l": "low"})

            for col in ("close", "high", "low"):
                if col not in df.columns:
                    logger.warning("Historical data missing required column: %s", col)
                    return None
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["close", "high", "low"])
            return df[["close", "high", "low"]]

        except asyncio.TimeoutError:
            logger.error("Historical data fetch timed out for %s", self.target_symbols[0])
            return None
        except Exception as exc:
            logger.error("Error fetching historical data: %s", exc, exc_info=True)
            return None

    async def on_model_update(self, event: Event) -> None:
        try:
            payload = event.payload or {}
            asset = payload.get("asset")
            new_rmse = float(payload.get("new_rmse", 0.0) or 0.0)
            confidence = float(payload.get("confidence_score", 0.0) or 0.0)

            logger.info(
                "[MODEL_UPDATE] Asset=%s New_RMSE=%.6f Confidence=%.4f",
                asset,
                new_rmse,
                confidence,
            )

            if asset:
                self._ml_confidence[asset] = confidence

        except Exception as exc:
            logger.error("Error in on_model_update: %s", exc, exc_info=True)

    async def _ml_retraining_loop(self) -> None:
        while not self._shutdown_started:
            try:
                await asyncio.sleep(float(os.getenv("HQC_ML_RETRAIN_SEC", "3600.0")))

                for symbol in self.target_symbols:
                    if symbol not in self._recent_ohlcv:
                        continue

                    ohlcv = self._recent_ohlcv[symbol]
                    if len(ohlcv["close"]) < 500:
                        logger.debug("[%s] Insufficient data for ML retraining", symbol)
                        continue

                    import pandas as pd

                    df = pd.DataFrame(
                        {
                            "open": ohlcv["open"],
                            "high": ohlcv["high"],
                            "low": ohlcv["low"],
                            "close": ohlcv["close"],
                            "volume": ohlcv["volume"],
                        }
                    )

                    logger.info("[%s] Triggering ML retraining...", symbol)
                    await self.ml_retrainer[symbol].trigger_retraining(df)

            except Exception as exc:
                logger.error("Error in ML retraining loop: %s", exc, exc_info=True)
                await asyncio.sleep(60.0)

    async def _health_loop(self) -> None:
        while self.state_machine.current_state != SystemState.HALTED and not self._shutdown_started:
            try:
                feed_ok = 1 if (self.data_feed._ws_task is not None and not self.data_feed._ws_task.done()) else 0
                router_ok = 1

                self.feedback.write_health(
                    state=self.state_machine.current_state.name,
                    status="running",
                    feed_ok=feed_ok,
                    router_ok=router_ok,
                    last_tick_symbol=self._last_tick_symbol,
                    last_tick_ts=self._last_tick_ts,
                    strategies_loaded=len(self.strategies),
                    extra={
                        "target_symbols": self.target_symbols,
                        "regime_classifier_fitted": self._regime_classifier_fitted,
                        "ml_models_active": len([v for v in self.ml_retrainer.values() if v.active_model is not None]),
                        "simulate_only": self.simulate_only,
                        "market_data_feed": self.market_data_feed,
                        "open_positions": self.eod_liquidator.snapshot(),
                    },
                )
                await asyncio.sleep(15.0)
            except Exception as exc:
                logger.error("Exception in health loop: %s", exc, exc_info=True)
                await asyncio.sleep(15.0)

    async def start(self) -> None:
        await self._wait_for_session_window()

        logger.info("=== INITIALIZING HYBRID QUANTITATIVE SYSTEM ===")
        logger.info("Target Universe: %s", self.target_symbols)
        logger.info("Execution Mode: %s", "PAPER" if self.is_paper else "LIVE STRICT")
        logger.info("Simulation Router: %s", "ENABLED" if self.simulate_only else "DISABLED")
        logger.info("Data Feed: ALPACA %s", self.market_data_feed.upper())
        logger.info("Learning: HMM Regime Classifier + ML Retraining Enabled")

        await self.bus.start()
        await self.slippage_controller.start()
        await self.execution_router.start()

        try:
            await self._reconcile_broker_state()
        except Exception as exc:
            logger.error("Broker reconciliation failed: %s", exc, exc_info=True)
            self.state_machine.transition_to(
                SystemState.HALTED,
                f"Broker reconciliation failed: {exc}",
            )
            await self.shutdown()
            return

        self.state_machine.transition_to(SystemState.WARMING_UP, "Services initialized.")
        self._warming_up_start_time = asyncio.get_event_loop().time()

        try:
            await self.data_feed.start()
            logger.info("Data feed connected successfully.")
        except RuntimeError as exc:
            logger.error("Data feed failed to initialize: %s", exc)
            self.state_machine.transition_to(
                SystemState.HALTED,
                f"Data feed initialization failed: {exc}",
            )
            await self.shutdown()
            return

        self._feed_task = asyncio.create_task(
            self._supervise_feed_connection(),
            name="feed_supervisor",
        )
        self._stale_check_task = asyncio.create_task(
            self._check_feed_staleness(),
            name="feed_staleness_monitor",
        )
        self._warming_up_check_task = asyncio.create_task(
            self._check_warming_up_timeout(),
            name="warming_up_timeout_monitor",
        )
        self._session_monitor_task = asyncio.create_task(
            self._session_window_monitor(),
            name="session_window_monitor",
        )
        self._health_task = asyncio.create_task(
            self._health_loop(),
            name="feedback_health_loop",
        )
        self._ml_retrain_task = asyncio.create_task(
            self._ml_retraining_loop(),
            name="ml_retraining_loop",
        )

        logger.info("Data feed task created. System remains in WARMING_UP until first live tick is received.")
        logger.info("=== SYSTEM ONLINE AND AWAITING MARKET TICKS ===")

        try:
            while self.state_machine.current_state != SystemState.HALTED:
                if self._feed_task is not None and self._feed_task.done():
                    exc = None
                    try:
                        exc = self._feed_task.exception()
                    except asyncio.CancelledError:
                        logger.info("Feed supervisor task was cancelled.")
                        break
                    except Exception as inner_exc:
                        logger.error("Error retrieving feed supervisor exception: %s", inner_exc, exc_info=True)

                    if exc is not None:
                        logger.error("Feed supervisor detected error: %s", exc, exc_info=True)
                        self.state_machine.transition_to(
                            SystemState.HALTED,
                            f"Feed supervisor error: {exc}",
                        )
                        break

                    logger.warning("Feed supervisor task exited unexpectedly.")
                    self.state_machine.transition_to(
                        SystemState.HALTED,
                        "Feed supervisor task exited unexpectedly.",
                    )
                    break

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.warning("TradingNode.start() cancelled.")
            raise
        finally:
            await self.shutdown()

    async def _supervise_feed_connection(self) -> None:
        logger.info("Feed supervisor started.")

        while not self._shutdown_started:
            if self.data_feed._ws_task is None:
                logger.error("Feed listen task not found!")
                await asyncio.sleep(1.0)
                continue

            if self.data_feed._ws_task.done():
                try:
                    exc = self.data_feed._ws_task.exception()
                except asyncio.CancelledError:
                    logger.info("Feed listen task was cancelled.")
                    break
                except Exception as exc:
                    logger.error("Error retrieving feed exception: %s", exc, exc_info=True)
                    raise

                if exc is not None:
                    logger.error("Feed listen task failed: %s", exc, exc_info=True)
                    raise exc

                logger.warning("Feed listen task exited without exception.")
                raise RuntimeError("Feed listen task exited unexpectedly")

            await asyncio.sleep(2.0)

    async def shutdown(self) -> None:
        if self._shutdown_started:
            return

        self._shutdown_started = True
        logger.warning("=== INITIATING GRACEFUL SHUTDOWN ===")

        self.feedback.write_health(
            state=self.state_machine.current_state.name,
            status="stopping",
            feed_ok=0,
            router_ok=0,
            last_tick_symbol=self._last_tick_symbol,
            last_tick_ts=self._last_tick_ts,
            strategies_loaded=len(self.strategies),
        )

        with contextlib.suppress(Exception):
            await self.data_feed.stop()

        if self._feed_task is not None:
            self._feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._feed_task

        if self._stale_check_task is not None:
            self._stale_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stale_check_task

        if self._warming_up_check_task is not None:
            self._warming_up_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._warming_up_check_task

        if self._session_monitor_task is not None:
            self._session_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._session_monitor_task

        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._health_task

        if self._ml_retrain_task is not None:
            self._ml_retrain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._ml_retrain_task

        with contextlib.suppress(Exception):
            await self.eod_liquidator.force_liquidate_now(reason="shutdown")
            await asyncio.sleep(0.1)

        with contextlib.suppress(Exception):
            await self.execution_router.stop()

        with contextlib.suppress(Exception):
            await self.slippage_controller.stop()

        with contextlib.suppress(Exception):
            await self.bus.stop()

        self.feedback.write_health(
            state="OFFLINE",
            status="stopped",
            feed_ok=0,
            router_ok=0,
            last_tick_symbol=self._last_tick_symbol,
            last_tick_ts=self._last_tick_ts,
            strategies_loaded=len(self.strategies),
        )

        logger.info("=== SYSTEM OFFLINE ===")

    @staticmethod
    def _parse_timestamp(ts_raw: object) -> int:
        if ts_raw is None:
            return int(time.time() * 1000)

        if isinstance(ts_raw, (int, float)):
            ts = int(ts_raw)
            return ts if ts > 10_000_000_000 else ts * 1000

        if isinstance(ts_raw, str):
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception:
                pass
            try:
                ts = int(float(ts_raw))
                return ts if ts > 10_000_000_000 else ts * 1000
            except Exception:
                return int(time.time() * 1000)

        return int(time.time() * 1000)


async def run_trading_node_once() -> None:
    node = TradingNode()
    try:
        await node.start()
    finally:
        with contextlib.suppress(Exception):
            await node.shutdown()


async def run_trading_node_forever() -> None:
    restart_backoff_sec = float(os.getenv("HQC_FOREVER_RESTART_BACKOFF_SEC", "15"))
    cycle_count = 0

    while True:
        cycle_count += 1
        logger.info("Starting trading cycle %d", cycle_count)
        try:
            await run_trading_node_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.critical("Trading cycle %d failed: %s", cycle_count, exc, exc_info=True)
            await asyncio.sleep(restart_backoff_sec)
            continue

        logger.info("Trading cycle %d completed. Preparing next session.", cycle_count)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop()

    def handle_exception(loop, context):
        msg = context.get("exception", context.get("message"))
        logger.critical("Caught Unhandled Exception: %s", msg)

    loop.set_exception_handler(handle_exception)

    try:
        run_forever = str(os.getenv("HQC_RUN_FOREVER", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if run_forever:
            loop.run_until_complete(run_trading_node_forever())
        else:
            loop.run_until_complete(run_trading_node_once())
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt (Ctrl+C) detected.")
    finally:
        loop.close()
        logger.info("Event loop closed. Process terminated.")