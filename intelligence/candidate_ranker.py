from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Deque, Dict, Optional, Set

import pytz

from core.engine.event_bus import Event, EventBus, EventType
from core.feedback.unified_logger import UnifiedFeedbackLogger
from intelligence.liquidity_rs_engine import LiquidityRelativeStrengthEngine

logger = logging.getLogger("CandidateRanker")


@dataclass
class TickSnapshot:
    price: float
    volume: float
    ts_ms: int


class RollingFeatureBook:
    """
    Minimal intraday feature state used by the ranker.
    Retained for time-of-day tick awareness and lightweight fallback state.
    """

    def __init__(self, benchmark: str = "SPY", tick_window: int = 300):
        self.benchmark = benchmark
        self.tick_window = tick_window
        self.ticks: Dict[str, Deque[TickSnapshot]] = defaultdict(lambda: deque(maxlen=tick_window))
        self.cum_pv: Dict[str, float] = defaultdict(float)
        self.cum_vol: Dict[str, float] = defaultdict(float)
        self.open_price: Dict[str, float] = {}
        self.last_price: Dict[str, float] = {}

    def update(self, symbol: str, price: float, volume: float, ts_ms: int) -> None:
        self.ticks[symbol].append(TickSnapshot(price=price, volume=volume, ts_ms=ts_ms))
        self.last_price[symbol] = price
        if symbol not in self.open_price:
            self.open_price[symbol] = price
        self.cum_pv[symbol] += price * volume
        self.cum_vol[symbol] += volume

    def reset_daily(self) -> None:
        self.ticks.clear()
        self.cum_pv.clear()
        self.cum_vol.clear()
        self.open_price.clear()
        self.last_price.clear()

    def vwap(self, symbol: str) -> Optional[float]:
        vol = self.cum_vol.get(symbol, 0.0)
        if vol <= 0:
            return None
        return self.cum_pv[symbol] / vol

    def rel_volume(self, symbol: str) -> float:
        ticks = self.ticks.get(symbol)
        if not ticks:
            return 1.0
        vols = [t.volume for t in ticks]
        if not vols:
            return 1.0
        avg_vol = sum(vols) / len(vols)
        if avg_vol <= 0:
            return 1.0
        return vols[-1] / avg_vol

    def spread_proxy_bps(self, symbol: str) -> float:
        ticks = self.ticks.get(symbol)
        if ticks is None or len(ticks) < 5:
            return 0.0
        prices = [t.price for t in ticks][-5:]
        hi, lo = max(prices), min(prices)
        mid = prices[-1]
        if mid <= 0:
            return 0.0
        return ((hi - lo) / mid) * 10000.0

    def rs_vs_benchmark(self, symbol: str) -> float:
        if symbol == self.benchmark:
            return 0.0
        symbol_open = self.open_price.get(symbol)
        bench_open = self.open_price.get(self.benchmark)
        symbol_last = self.last_price.get(symbol)
        bench_last = self.last_price.get(self.benchmark)
        if not all([symbol_open, bench_open, symbol_last, bench_last]):
            return 0.0
        sym_ret = (symbol_last / symbol_open) - 1.0
        bench_ret = (bench_last / bench_open) - 1.0
        return sym_ret - bench_ret


class CandidateRanker:
    """
    Filters and scores raw strategy intents before risk sizing.

    Design constraint:
    - Existing stack only exposes EventType.ORDER_CREATE for strategy->risk->router flow.
    - We therefore use payload stage flags instead of introducing a new enum dependency.
    
    FIX: Properly handle nested dict structure from LiquidityRelativeStrengthEngine.evaluate_candidate()
    FIX: Deduplicate signals using signal_id with 2-second rolling window
    """

    def __init__(
        self,
        bus: EventBus,
        benchmark: str = "SPY",
        min_score: float = 4.75,
        max_spread_bps: float = 18.0,
        max_dist_vwap_pct: float = 0.012,
        decisions_path: str = "state/feedback/decisions.jsonl",
        dedup_window_sec: float = 2.0,  # --- FIX: Deduplication window ---
        ml_gate: object | None = None,
    ) -> None:
        self.bus = bus
        # Optional ML probability gate. Backtest-only veto layer; if
        # provided, the ranker will consult it after the rule-based score
        # passes and before publishing the RANKED event. The gate must
        # expose a synchronous should_pass(symbol, action, ts_ms) method
        # that returns an object with `.passed` (bool), `.probability`,
        # and `.reason` attributes.
        self.ml_gate = ml_gate
        self.book = RollingFeatureBook(benchmark=benchmark)
        self.liq_rs = LiquidityRelativeStrengthEngine(
            benchmark=benchmark,
            tick_window=300,
            min_ticks_for_quality=8,
            max_spread_bps=max_spread_bps,
            min_rvol=0.90,
            min_liquidity_score=0.35,
        )

        self.benchmark = benchmark
        self.min_score = min_score
        self.max_spread_bps = max_spread_bps
        self.max_dist_vwap_pct = max_dist_vwap_pct
        
        # --- FIX: Track processed signal IDs and timestamps ---
        self.processed_signal_ids: Dict[str, float] = {}  # signal_id -> timestamp_ms
        self.dedup_window_ms = dedup_window_sec * 1000.0
        # --- END FIX ---

        self.decisions_path = Path(decisions_path)
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)

        self.feedback = UnifiedFeedbackLogger(
            root=str(self.decisions_path.parent),
            system_name="HQC",
            arm="equities",
            env=os.getenv("TRADING_MODE", "PAPER"),
        )

        self._tz = pytz.timezone("US/Eastern")
        self._market_open = time(9, 30)
        self._lunch_start = time(11, 45)
        self._lunch_end = time(13, 15)
        self._current_trading_date: Optional[date] = None

        self.bus.subscribe(EventType.TICK, self.on_tick)
        self.bus.subscribe(EventType.ORDER_CREATE, self.on_order_create)

    async def on_tick(self, event: Event) -> None:
        payload = event.payload or {}
        symbol = payload.get("ticker") or payload.get("symbol")
        price = payload.get("price")
        volume = payload.get("volume")
        ts_raw = payload.get("timestamp")

        if not symbol or price is None or volume is None:
            return

        symbol = str(symbol).upper()
        ts_ms = self._normalize_ts_ms(ts_raw)
        tick_dt_est = datetime.fromtimestamp(ts_ms / 1000.0, tz=pytz.utc).astimezone(self._tz)
        tick_date = tick_dt_est.date()
        if self._current_trading_date != tick_date:
            self.book.reset_daily()
            self.liq_rs.reset_daily()
            self.processed_signal_ids.clear()
            self._current_trading_date = tick_date
            logger.info("[%s] Daily state reset completed for %s", self.__class__.__name__, tick_date)

        self.book.update(
            symbol=symbol,
            price=float(price),
            volume=float(volume),
            ts_ms=ts_ms,
        )
        self.liq_rs.update_tick(
            symbol=symbol,
            price=float(price),
            volume=float(volume),
            ts_ms=ts_ms,
        )

    async def on_order_create(self, event: Event) -> None:
        payload = dict(event.payload or {})

        # Only process raw strategy intents once.
        if payload.get("stage") in {"RANKED", "SIZED", "ROUTED"}:
            return
        if payload.get("shares") is not None:
            return

        symbol = payload.get("asset")
        action = str(payload.get("action", "")).upper()
        entry_price = payload.get("reference_price")
        strategy = payload.get("strategy", "Unknown")
        
        # --- FIX: Extract signal_id for deduplication ---
        signal_id = payload.get("signal_id")
        # Use the strategy's event timestamp when available so dedup is
        # immune to wall-clock drift between event publish and ranker
        # processing; fall back to "now" when the strategy did not stamp it.
        ts_ms = self._normalize_ts_ms(payload.get("timestamp"))
        
        if not self._is_duplicate_signal(signal_id, ts_ms):
            logger.debug(
                "[%s] Processing new signal (signal_id=%s)",
                symbol,
                signal_id[:8] if signal_id else "none",
            )
        else:
            logger.warning(
                "[%s] Duplicate signal detected (signal_id=%s). Ignoring.",
                symbol,
                signal_id[:8] if signal_id else "none",
            )
            return
        # --- END FIX ---

        if not symbol or not action or entry_price is None:
            return

        symbol = str(symbol).upper()
        entry_price = float(entry_price)

        try:
            score_card = self._score_candidate(
                symbol=symbol,
                action=action,
                entry_price=entry_price,
            )
        except Exception as e:
            logger.error(
                "Failed to score candidate %s %s @ %.2f: %s",
                symbol,
                action,
                entry_price,
                e,
                exc_info=True,
            )
            return

        decision_id = str(uuid.uuid4())
        approved = score_card["score"] >= self.min_score and not score_card["hard_veto"]

        ml_gate_meta: Dict[str, object] = {}
        if approved and self.ml_gate is not None:
            try:
                gate_decision = self.ml_gate.should_pass(symbol=symbol, action=action, ts_ms=ts_ms)
            except Exception as exc:
                logger.warning(
                    "[%s] ML gate raised; treating as abstain. err=%s",
                    symbol,
                    exc,
                )
                gate_decision = None
            if gate_decision is not None:
                ml_gate_meta = {
                    "ml_pass": bool(getattr(gate_decision, "passed", True)),
                    "ml_probability": float(getattr(gate_decision, "probability", 0.5)),
                    "ml_threshold": float(getattr(gate_decision, "threshold", 0.5)),
                    "ml_reason": str(getattr(gate_decision, "reason", "")),
                }
                if not ml_gate_meta["ml_pass"]:
                    approved = False
                    score_card.setdefault("reasons", []).append("ml_gate_veto")

        decision_payload = {
            **payload,
            "decision_id": decision_id,
            "stage": "RANKED",
            "approved_by_ranker": approved,
            "rank_score": score_card["score"],
            "rank_components": score_card,
            "ml_gate": ml_gate_meta,
        }

        self.feedback.write_decision(
            {
                "decision_id": decision_id,
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "symbol": symbol,
                "strategy": strategy,
                "side": action,
                "entry_price": entry_price,
                "approved": approved,
                "score": score_card["score"],
                "components": score_card,
                "meta": {
                    "stage": "RANKED",
                    "approved_by_ranker": approved,
                    "signal_id": signal_id,  # --- FIX: Track signal_id in decision ---
                },
            }
        )

        status = "PASS" if approved else "BLOCK"
        logger.info(
            "[%s] %s %s | scr:%.2f veto:%s | rs:%.2f rvol:%.1f vwap:%.3f spr:%.1f | signal_id=%s",
            status,
            symbol,
            action,
            score_card["score"],
            score_card["hard_veto"],
            score_card["rs"],
            score_card["rvol"],
            score_card["dist_vwap_pct"],
            score_card["spread_bps"],
            signal_id[:8] if signal_id else "none",  # --- FIX: Log signal_id ---
        )
        
        if not approved and score_card["reasons"]:
            logger.info("       -> Veto Reasons: %s", score_card["reasons"])

        if approved:
            self.bus.publish(Event(type=EventType.ORDER_CREATE, payload=decision_payload))

    def _is_duplicate_signal(self, signal_id: Optional[str], ts_ms: int) -> bool:
        """
        --- FIX: Check if signal_id has been processed recently ---
        
        Prevents duplicate processing within dedup_window_ms (default 2 seconds).
        """
        if not signal_id:
            return False  # No signal_id, allow (shouldn't happen with new code)
        
        # Clean up old entries outside the window
        current_time_ms = ts_ms
        self.processed_signal_ids = {
            sig_id: ts for sig_id, ts in self.processed_signal_ids.items()
            if (current_time_ms - ts) < self.dedup_window_ms
        }
        
        # Check if this signal_id was seen recently
        if signal_id in self.processed_signal_ids:
            age_ms = current_time_ms - self.processed_signal_ids[signal_id]
            logger.debug(
                "Signal %s seen %.0f ms ago (window: %.0f ms)",
                signal_id[:8],
                age_ms,
                self.dedup_window_ms,
            )
            return True
        
        # Mark this signal_id as processed
        self.processed_signal_ids[signal_id] = current_time_ms
        return False
        # --- END FIX ---

    def _score_candidate(self, symbol: str, action: str, entry_price: float) -> Dict[str, object]:
        """
        Score a candidate trade.
        
        FIX: Properly extract metrics from nested structure returned by evaluate_candidate()
        """
        eval_result = self.liq_rs.evaluate_candidate(
            symbol=symbol,
            action=action,
            reference_price=entry_price,
        )
        
        metrics = eval_result.get("metrics", {})
        scoring = eval_result.get("scoring", {})
        veto = eval_result.get("veto", {})
        context = eval_result.get("context", {})

        rs = float(metrics.get("rs", 0.0))
        rvol = float(metrics.get("rvol", 1.0))
        spread_bps = float(metrics.get("spread_bps", 0.0))
        dist_vwap_pct = float(metrics.get("dist_vwap_pct", 0.0))
        liquidity_score = float(metrics.get("liquidity_score", 0.0))
        
        hard_veto = bool(veto.get("hard_veto", False))
        reasons = list(veto.get("reasons", []))

        # Treat low relative volume as a hard veto at ranker gate to avoid thin-tape entries.
        if rvol < self.liq_rs.min_rvol:
            hard_veto = True
            rvol_reason = f"rvol<{self.liq_rs.min_rvol:.2f}"
            if rvol_reason not in reasons:
                reasons.append(rvol_reason)

        base_score = float(scoring.get("total_score", 0.0))

        tod_mult = self._time_of_day_multiplier(symbol)
        
        is_aligned_with_vwap = (action == "BUY" and dist_vwap_pct >= 0) or (action in {"SELL", "SELL_SHORT"} and dist_vwap_pct <= 0)
        
        vwap_penalty = 1.0
        if not is_aligned_with_vwap:
            vwap_penalty = 0.85
            reasons.append("fighting_vwap_trend")

        if abs(dist_vwap_pct) > self.max_dist_vwap_pct:
            reasons.append(f"overextended_vwap: {dist_vwap_pct:.4f}")
            hard_veto = True

        final_score = base_score * tod_mult * vwap_penalty

        return {
            "rs": round(rs, 6),
            "rvol": round(rvol, 4),
            "spread_bps": round(spread_bps, 4),
            "dist_vwap_pct": round(dist_vwap_pct, 6),
            "liquidity_score": round(liquidity_score, 4),
            "time_of_day_mult": round(tod_mult, 4),
            "vwap_alignment_mult": round(vwap_penalty, 2),
            "score": round(max(0.0, final_score), 4),
            "confidence_metric": round(min(1.0, final_score / 10.0), 4),
            "hard_veto": hard_veto,
            "reasons": reasons,
        }

    def _time_of_day_multiplier(self, symbol: str) -> float:
        ticks = self.book.ticks.get(symbol)
        if not ticks:
            return 1.0

        dt_est = datetime.fromtimestamp(
            ticks[-1].ts_ms / 1000.0,
            tz=pytz.utc,
        ).astimezone(self._tz)

        t = dt_est.time()

        if self._market_open <= t <= time(10, 30):
            minutes_since_open = (t.hour - 9) * 60 + (t.minute - 30)
            decay_mult = 1.20 - (0.20 * (minutes_since_open / 60.0))
            return max(1.0, decay_mult)
            
        if self._lunch_start <= t <= self._lunch_end:
            return 0.82
            
        if time(15, 0) <= t <= time(15, 50):
            return 1.05
            
        return 1.0

    @staticmethod
    def _normalize_ts_ms(ts_raw: object) -> int:
        if ts_raw is None:
            return int(datetime.now(timezone.utc).timestamp() * 1000)

        if isinstance(ts_raw, (int, float)):
            ts = int(ts_raw)
            return ts if ts > 10_000_000_000 else ts * 1000

        if isinstance(ts_raw, str):
            try:
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception:
                pass
            try:
                ts = int(float(ts_raw))
                return ts if ts > 10_000_000_000 else ts * 1000
            except Exception:
                return int(datetime.now(timezone.utc).timestamp() * 1000)

        return int(datetime.now(timezone.utc).timestamp() * 1000)