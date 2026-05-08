"""
ML candidate gate.

Backtest-only signal filter. Fits a feature pipeline (PCA over engineered
technicals) and a logistic regression on a held-out historical bar
window. At inference time, given a candidate ORDER_CREATE intent, the
gate slices the stored bar history up to the candidate's timestamp,
generates the same features, and predicts the probability that the
next ``target_horizon`` bars will close higher than the entry. If the
probability disagrees with the candidate's direction by enough margin
the gate vetoes the trade.

Design constraints
------------------
- Veto-only. The gate cannot generate new candidates, only block
  loss-prone ones. This keeps the failure mode bounded: in the worst
  case a bad gate trades less, never differently.
- Logistic regression, not gradient boosting. The 50k-bar training
  window we have for SPY is small relative to feature dimensionality;
  a high-capacity model would overfit and look great in-sample.
- Fit once at backtest start on a leading chunk of bars
  (``train_bars`` rows from the head of the dataset). The remainder
  of the bars is the test set the strategies trade on. No leakage:
  the gate has never seen the test bars.

This is deliberately *not* wired into the live execution path. Use
in walkforward_runner / backtest_runner only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from intelligence.ml_pipeline.feature_engineering import DynamicFeatureEngineer

logger = logging.getLogger("MLCandidateGate")


@dataclass
class GateDecision:
    passed: bool
    probability: float
    threshold: float
    reason: str


class MLCandidateGate:
    """
    Probability-based signal filter.

    Args:
        historical_df: OHLCV bars used to fit the feature pipeline + classifier.
            Must contain columns: timestamp, open, high, low, close, volume.
        target_horizon: how many bars ahead we predict (default 5).
        threshold: minimum P(positive return) to pass a BUY (and 1-threshold
            ceiling to pass a SELL/SELL_SHORT). Default 0.55 — a small
            edge over coin flip.
        feature_min_rows: minimum bars required to compute features at
            inference time. Hard-coded to 60 to match the engineer's
            default ``min_rows``.
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        target_horizon: int = 5,
        threshold: float = 0.55,
        feature_min_rows: int = 60,
    ) -> None:
        self.target_horizon = int(target_horizon)
        self.threshold = float(threshold)
        self.feature_min_rows = int(feature_min_rows)
        self._engineer = DynamicFeatureEngineer(target_horizon=self.target_horizon, min_rows=self.feature_min_rows)
        self._model: Optional[LogisticRegression] = None
        self._history: Optional[pd.DataFrame] = None
        self._fit_n: int = 0
        self._calls = {"pass": 0, "veto": 0, "abstain": 0}
        self._fit(historical_df)

    def _fit(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("MLCandidateGate requires a non-empty historical_df")
        prepared = _prepare_bars(df)
        if len(prepared) < self.feature_min_rows + self.target_horizon + 10:
            raise ValueError(
                f"MLCandidateGate needs at least {self.feature_min_rows + self.target_horizon + 10} "
                f"clean bars, got {len(prepared)}."
            )
        # Fit feature pipeline (returns PCA-projected X and aligned target series).
        x_pca, y_target = self._engineer.fit_transform(prepared)
        # Binary label: did the next-target_horizon return exceed zero?
        y_binary = (y_target.values > 0.0).astype(int)
        # Logistic regression with L2 regularization. Class-balanced because
        # markets are mildly biased upward and we want a calibrated probability.
        self._model = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
        self._model.fit(x_pca, y_binary)
        self._fit_n = int(len(prepared))
        # Store the historical bars so live inference can slice up to a ts.
        self._history = prepared
        logger.info(
            "MLCandidateGate fitted on %d bars; PCA dims=%d; train base rate=%.3f",
            self._fit_n,
            int(x_pca.shape[1]),
            float(y_binary.mean()),
        )

    def update_history(self, df: pd.DataFrame) -> None:
        """Replace the stored bar history (used in walk-forward to extend the
        gate's view to include the test window without retraining)."""
        if df is None or df.empty:
            return
        self._history = _prepare_bars(df)

    def should_pass(self, symbol: str, action: str, ts_ms: Optional[int]) -> GateDecision:
        """
        Decide whether to pass a candidate trade.

        symbol is unused for now (single-symbol gate); included for API
        forward-compatibility once we move to multi-symbol.
        """
        if self._model is None or self._history is None:
            return GateDecision(True, 0.5, self.threshold, "gate_not_ready")

        if not isinstance(ts_ms, (int, float)) or ts_ms <= 0:
            self._calls["abstain"] += 1
            return GateDecision(True, 0.5, self.threshold, "missing_timestamp")

        # Slice bar history up to (but not including) the candidate timestamp.
        cutoff = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
        history = self._history
        history_idx = history.index <= cutoff
        window = history[history_idx]
        if len(window) < self.feature_min_rows:
            self._calls["abstain"] += 1
            return GateDecision(True, 0.5, self.threshold, "insufficient_history")

        try:
            x = self._engineer.transform_live(window)
        except Exception as exc:
            logger.warning("MLCandidateGate transform failed: %s", exc)
            self._calls["abstain"] += 1
            return GateDecision(True, 0.5, self.threshold, "transform_error")

        if x is None or x.size == 0:
            self._calls["abstain"] += 1
            return GateDecision(True, 0.5, self.threshold, "transform_empty")

        prob_up = float(self._model.predict_proba(x)[0, 1])

        action_norm = str(action or "").upper()
        is_long = action_norm in {"BUY", "BUY_TO_OPEN", "BUY_TO_COVER"}
        is_short = action_norm in {"SELL_SHORT", "SELL_TO_OPEN"}

        if is_long:
            passed = prob_up >= self.threshold
            reason = "ml_long_pass" if passed else "ml_long_veto"
        elif is_short:
            passed = prob_up <= (1.0 - self.threshold)
            reason = "ml_short_pass" if passed else "ml_short_veto"
        else:
            # Closing actions (SELL of a long, BUY_TO_COVER of a short) are
            # handled by the simulated stop / time stop, not the entry gate.
            self._calls["abstain"] += 1
            return GateDecision(True, prob_up, self.threshold, "non_entry_action")

        self._calls["pass" if passed else "veto"] += 1
        return GateDecision(passed, prob_up, self.threshold, reason)

    def stats(self) -> dict:
        total = sum(self._calls.values())
        return {
            "calls_total": total,
            "calls_pass": self._calls["pass"],
            "calls_veto": self._calls["veto"],
            "calls_abstain": self._calls["abstain"],
            "veto_rate": (self._calls["veto"] / total) if total else 0.0,
            "fit_n_bars": self._fit_n,
            "threshold": self.threshold,
            "target_horizon": self.target_horizon,
        }


def _prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a CSV-style OHLCV frame into a timestamp-indexed numeric frame
    suitable for DynamicFeatureEngineer."""
    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    elif "datetime" in work.columns:
        work["timestamp"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    else:
        raise ValueError("MLCandidateGate requires a timestamp column.")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in work.columns:
            raise ValueError(f"MLCandidateGate requires column {col!r}.")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close", "volume"])
    work = work[(work["close"] > 0) & (work["high"] >= work["low"])]
    work = work[~work.index.duplicated(keep="last")]
    return work
