from __future__ import annotations

import json
import logging
import socket
import threading
import queue
import atexit
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

logger = logging.getLogger("UnifiedFeedbackLogger")


class UnifiedFeedbackLogger:
    """
    Omega unified feedback logging standard (append-only JSONL).

    Upgrades:
    - Dedicated background I/O thread.
    - Non-blocking queue ingestion for strict async determinism.
    - Automatic graceful shutdown via atexit.

    Files:
      state/feedback/decisions.jsonl
      state/feedback/outcomes.jsonl
      state/feedback/health.jsonl
    """

    def __init__(
        self,
        root: str = "state/feedback",
        system_name: str = "HQC",
        arm: str = "equities",
        env: str = "paper",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.system_name = str(system_name)
        self.arm = str(arm)
        self.env = str(env).lower()

        self.decisions_path = self.root / "decisions.jsonl"
        self.outcomes_path = self.root / "outcomes.jsonl"
        self.health_path = self.root / "health.jsonl"

        self.hostname = socket.gethostname()

        # Non-blocking I/O Architecture with persistent file handles + batched flush.
        # Keeping handles open avoids per-line open/close syscalls; batched flush
        # bounds disk-flush frequency rather than fsyncing on every record.
        self._log_queue: queue.Queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._file_handles: Dict[Path, TextIO] = {}
        self._flush_interval_sec: float = 0.5
        self._max_batch: int = 256

        # Daemon thread ensures it doesn't prevent the main program from exiting
        self._worker_thread = threading.Thread(target=self._io_worker, name="LoggerIOWorker", daemon=True)
        self._worker_thread.start()

        # Ensure we flush the queue when the bot shuts down
        atexit.register(self.shutdown)

    def _get_handle(self, path: Path) -> TextIO:
        handle = self._file_handles.get(path)
        if handle is None or handle.closed:
            handle = path.open("a", encoding="utf-8", buffering=1024 * 64)
            self._file_handles[path] = handle
        return handle

    def _io_worker(self) -> None:
        """Background thread that pops logs from memory and writes to disk."""
        last_flush = time.monotonic()
        dirty_handles: set[TextIO] = set()

        while not self._shutdown_event.is_set() or not self._log_queue.empty():
            try:
                path, payload = self._log_queue.get(timeout=self._flush_interval_sec)
            except queue.Empty:
                # Idle — flush any pending writes and continue
                if dirty_handles:
                    for h in dirty_handles:
                        try:
                            h.flush()
                        except Exception as exc:
                            logger.error("Flush failure in UnifiedFeedbackLogger: %s", exc)
                    dirty_handles.clear()
                    last_flush = time.monotonic()
                continue

            try:
                line = json.dumps(payload, separators=(",", ":"), default=str) + "\n"
                handle = self._get_handle(path)
                handle.write(line)
                dirty_handles.add(handle)
                self._log_queue.task_done()
            except Exception as e:
                logger.error("Background I/O failure in UnifiedFeedbackLogger: %s", e)
                continue

            # Drain a small batch before flushing to amortize the flush cost.
            drained = 0
            while drained < self._max_batch:
                try:
                    path, payload = self._log_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    line = json.dumps(payload, separators=(",", ":"), default=str) + "\n"
                    handle = self._get_handle(path)
                    handle.write(line)
                    dirty_handles.add(handle)
                    self._log_queue.task_done()
                except Exception as e:
                    logger.error("Background I/O failure in UnifiedFeedbackLogger: %s", e)
                drained += 1

            now = time.monotonic()
            if (now - last_flush) >= self._flush_interval_sec or drained >= self._max_batch:
                for h in dirty_handles:
                    try:
                        h.flush()
                    except Exception as exc:
                        logger.error("Flush failure in UnifiedFeedbackLogger: %s", exc)
                dirty_handles.clear()
                last_flush = now

        # Final drain on shutdown
        for h in dirty_handles:
            try:
                h.flush()
            except Exception:
                pass
        for h in list(self._file_handles.values()):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
        self._file_handles.clear()

    def shutdown(self) -> None:
        """Gracefully waits for remaining logs to write before terminating."""
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()
            if self._worker_thread.is_alive():
                self._worker_thread.join(timeout=3.0)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        # Puts the log item into memory instantly without blocking the async event loop
        self._log_queue.put((path, payload))

    def write_decision(self, payload: Dict[str, Any]) -> None:
        record = {
            "type": "decision",
            "ts": payload.get("ts", self._utc_now()),
            "system": self.system_name,
            "arm": self.arm,
            "env": self.env,
            "decision_id": payload.get("decision_id"),
            "symbol": payload.get("symbol"),
            "strategy": payload.get("strategy"),
            "side": payload.get("side"),
            "entry_price": payload.get("entry_price"),
            "approved": payload.get("approved"),
            "score": payload.get("score"),
            "components": payload.get("components") or {},
            "meta": payload.get("meta") or {},
        }
        self._append_jsonl(self.decisions_path, record)

    def write_outcome(self, payload: Dict[str, Any]) -> None:
        record = {
            "type": "outcome",
            "ts": payload.get("ts", self._utc_now()),
            "system": self.system_name,
            "arm": self.arm,
            "env": self.env,
            "decision_id": payload.get("decision_id"),
            "order_id": payload.get("order_id"),
            "status": payload.get("status"),
            "symbol": payload.get("symbol"),
            "strategy": payload.get("strategy"),
            "side": payload.get("side"),
            "qty": payload.get("qty"),
            "filled_qty": payload.get("filled_qty"),
            "entry_price": payload.get("entry_price"),
            "fill_price": payload.get("fill_price"),
            "exit_price": payload.get("exit_price"),
            "gross_pnl": payload.get("gross_pnl"),
            "net_pnl": payload.get("net_pnl"),
            "hold_seconds": payload.get("hold_seconds"),
            "win": payload.get("win"),
            "mfe": payload.get("mfe"),
            "mae": payload.get("mae"),
            "meta": payload.get("meta") or {},
        }
        self._append_jsonl(self.outcomes_path, record)

    def write_health(
        self,
        *,
        state: str,
        status: str,
        feed_ok: int,
        router_ok: int,
        last_tick_symbol: Optional[str] = None,
        last_tick_ts: Optional[str] = None,
        strategies_loaded: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "type": "health",
            "ts": self._utc_now(),
            "system": self.system_name,
            "arm": self.arm,
            "env": self.env,
            "hostname": self.hostname,
            "status": status,
            "state": state,
            "feed_ok": int(feed_ok),
            "router_ok": int(router_ok),
            "last_tick_symbol": last_tick_symbol,
            "last_tick_ts": last_tick_ts,
            "strategies_loaded": int(strategies_loaded),
            "meta": extra or {},
        }
        self._append_jsonl(self.health_path, record)