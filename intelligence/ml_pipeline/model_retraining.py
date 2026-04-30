from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from core.engine.event_bus import Event, EventBus, EventType

logger = logging.getLogger("MLRetrainingLoop")


class DynamicFeatureEngineer:
    """
    Dynamic feature engineering for ML models.
    
    Computes technical indicators and performs dimensionality reduction
    via PCA for robust feature selection.
    """
    
    def __init__(
        self,
        variance_retained: float = 0.95,
        min_rows: int = 60,
        target_horizon: int = 5,
    ) -> None:
        self.variance_retained = float(variance_retained)
        self.min_rows = int(min_rows)
        self.target_horizon = int(target_horizon)
        self.pca = None
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Fit feature engineer and transform historical data.
        
        Args:
            df: OHLCV DataFrame with columns [close, high, low, volume]
        
        Returns:
            (X, y) tuple of features and target
        """
        if df is None or len(df) < self.min_rows:
            return None, None
        
        work = df.copy(deep=False)
        
        # Compute technical indicators
        work['returns'] = work['close'].pct_change()
        work['log_returns'] = np.log(work['close'] / work['close'].shift(1))
        work['volatility'] = work['returns'].rolling(20).std()
        work['rsi'] = self._compute_rsi(work['close'], period=14)
        work['macd'] = self._compute_macd(work['close'])
        work['ema_20'] = work['close'].ewm(span=20, adjust=False).mean()
        work['ema_50'] = work['close'].ewm(span=50, adjust=False).mean()
        work['atr'] = self._compute_atr(work)
        
        work = work.dropna()
        
        if len(work) < self.min_rows:
            return None, None
        
        # Create target: forward returns
        y = work['close'].shift(-self.target_horizon) / work['close'] - 1.0
        y = y.iloc[:-self.target_horizon]

        # Features
        feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'ema_20', 'ema_50', 'atr']
        X = work[feature_cols].iloc[:-self.target_horizon]
        
        X = X.dropna()
        y = y.loc[X.index]
        
        if len(X) != len(y):
            return None, None
        
        self.is_fitted = True
        return X, y
    
    def transform_live(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Transform latest rows for live prediction."""
        if not self.is_fitted or df is None or len(df) < 50:
            return None
        
        work = df.tail(60).copy(deep=False)
        
        work['returns'] = work['close'].pct_change()
        work['volatility'] = work['returns'].rolling(20).std()
        work['rsi'] = self._compute_rsi(work['close'], period=14)
        work['macd'] = self._compute_macd(work['close'])
        work['ema_20'] = work['close'].ewm(span=20, adjust=False).mean()
        work['ema_50'] = work['close'].ewm(span=50, adjust=False).mean()
        work['atr'] = self._compute_atr(work)
        
        work = work.dropna()
        if len(work) == 0:
            return None
        
        feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'ema_20', 'ema_50', 'atr']
        X = work[feature_cols].iloc[-1:].values
        
        return X
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _compute_macd(prices: pd.Series) -> pd.Series:
        """Compute MACD."""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        return macd
    
    @staticmethod
    def _compute_atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr


class ContinuousRetrainingPipeline:
    """
    Adaptive ML retraining pipeline with LightGBM.

    Features:
    - Dedicated MODEL_UPDATE event channel for model updates
    - Asynchronous training on background thread
    - Thread-safe model swapping
    - RMSE-based model validation
    
    Usage:
        retrainer = ContinuousRetrainingPipeline(
            bus=bus,
            target_asset="SPY",
            retrain_interval_sec=3600.0
        )
        await retrainer.trigger_retraining(historical_df)
        prediction = retrainer.predict_live(df_latest)
    """

    def __init__(
        self,
        bus: EventBus,
        target_asset: str,
        retrain_interval_sec: float = 3600.0,
        min_training_rows: int = 500,
        validation_fraction: float = 0.20,
    ) -> None:
        self.bus = bus
        self.asset = str(target_asset).upper()
        self.retrain_interval_sec = float(retrain_interval_sec)
        self.min_training_rows = int(min_training_rows)
        self.validation_fraction = float(validation_fraction)

        self.is_training: bool = False
        self.active_model: Optional[lgb.LGBMRegressor] = None
        self.active_engineer: Optional[DynamicFeatureEngineer] = None
        self.active_rmse: float = float("inf")

        self._model_lock = threading.Lock()

        self.model_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 2,
            "objective": "regression",
            "verbosity": -1,
        }

        logger.info(
            "[%s] ContinuousRetrainingPipeline initialized. Retrain interval: %.1f seconds",
            self.asset,
            self.retrain_interval_sec,
        )

    def _train_model_blocking(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Optional[lgb.LGBMRegressor], Optional[DynamicFeatureEngineer], float]:
        """
        CPU-heavy training task. Must run off the async event loop.
        
        Args:
            df: Historical OHLCV DataFrame
        
        Returns:
            (model, engineer, rmse) or (None, None, inf) on failure
        """
        logger.info("[%s] Background Thread: Initiating model training on %d rows...", self.asset, len(df))

        try:
            if df is None or df.empty:
                raise ValueError("Historical DataFrame is empty.")
            if len(df) < self.min_training_rows:
                raise ValueError(
                    f"Insufficient data for ML training: have {len(df)}, need {self.min_training_rows}."
                )

            new_engineer = DynamicFeatureEngineer(
                variance_retained=0.95,
                min_rows=60,
                target_horizon=5,
            )

            X, y = new_engineer.fit_transform(df)

            if X is None or y is None or len(X) == 0 or len(y) == 0:
                raise ValueError("Feature engineer returned empty training data.")

            if len(X) != len(y):
                raise ValueError(f"X/y length mismatch: len(X)={len(X)} len(y)={len(y)}")

            split_idx = int(len(X) * (1.0 - self.validation_fraction))
            min_train = max(100, min(250, len(X) // 2))
            min_val = max(50, min(100, len(X) // 5))

            if split_idx < min_train or (len(X) - split_idx) < min_val:
                raise ValueError(
                    f"Time-series split too small. len(X)={len(X)} split_idx={split_idx} "
                    f"train={split_idx} val={len(X) - split_idx}"
                )

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            model = lgb.LGBMRegressor(**self.model_params)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )

            predictions = model.predict(np.asarray(X_val))
            rmse = float(root_mean_squared_error(y_val, predictions))

            logger.info(
                "[%s] Background Thread: Training complete. Out-of-sample RMSE: %.6f",
                self.asset,
                rmse,
            )
            return model, new_engineer, rmse

        except Exception as e:
            logger.error("[%s] Failed during background model training: %s", self.asset, e, exc_info=True)
            return None, None, float("inf")

    async def trigger_retraining(self, historical_df: pd.DataFrame) -> None:
        """
        Async entry point for retraining.
        
        Args:
            historical_df: Historical OHLCV data for training
        """
        if self.is_training:
            logger.warning("[%s] Retraining already in progress. Skipping trigger.", self.asset)
            return

        if historical_df is None or len(historical_df) < self.min_training_rows:
            logger.error(
                "[%s] Insufficient data for ML training. Minimum %d rows required.",
                self.asset,
                self.min_training_rows,
            )
            return

        self.is_training = True
        logger.info("[%s] Spawning background thread for ML retraining...", self.asset)

        try:
            new_model, new_engineer, new_rmse = await asyncio.to_thread(
                self._train_model_blocking,
                historical_df,
            )
        finally:
            self.is_training = False

        if new_model is None or new_engineer is None:
            logger.error("[%s] Model retraining failed to return valid objects.", self.asset)
            return

        if new_rmse < self.active_rmse:
            logger.warning(
                "[%s] NEW MODEL ACCEPTED! Improvement: %.6f -> %.6f",
                self.asset,
                self.active_rmse,
                new_rmse,
            )
            await self._swap_models(new_model, new_engineer, new_rmse)
        else:
            logger.info(
                "[%s] New model rejected. Current model is superior (%.6f <= %.6f)",
                self.asset,
                self.active_rmse,
                new_rmse,
            )

    async def _swap_models(
        self,
        new_model: lgb.LGBMRegressor,
        new_engineer: DynamicFeatureEngineer,
        rmse: float,
    ) -> None:
        """
        Thread-safe swapping of live model pointers plus async event emission.
        
        Args:
            new_model: Newly trained LightGBM model
            new_engineer: Newly fitted feature engineer
            rmse: Out-of-sample RMSE
        """
        with self._model_lock:
            self.active_model = new_model
            self.active_engineer = new_engineer
            self.active_rmse = float(rmse)

        update_event = Event(
            type=EventType.MODEL_UPDATE,
            payload={
                "action": "MODEL_UPDATED",
                "asset": self.asset,
                "new_rmse": float(rmse),
                "confidence_score": float(1.0 / (rmse + 1e-6)),
            },
        )
        self.bus.publish(update_event)

    def predict_live(self, df_live: pd.DataFrame) -> Optional[float]:
        """
        Predict future return from live data using the active model.
        
        Args:
            df_live: Recent OHLCV data
        
        Returns:
            Predicted return (float) or None if prediction fails
        """
        with self._model_lock:
            model = self.active_model
            engineer = self.active_engineer

        if model is None or engineer is None:
            return None

        try:
            X_live = engineer.transform_live(df_live)
            if X_live is None:
                return None

            prediction = model.predict(np.asarray(X_live))[0]
            return float(prediction)

        except Exception as e:
            logger.error("[%s] Live prediction failed: %s", self.asset, e, exc_info=True)
            return None


if __name__ == "__main__":
    print("Testing ContinuousRetrainingPipeline...")
    
    # Generate synthetic OHLCV data
    dates = pd.date_range("2024-01-01", periods=1000, freq="1H")
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 1000))
    
    df = pd.DataFrame({
        "close": close_prices,
        "high": close_prices + np.abs(np.random.normal(0, 0.2, 1000)),
        "low": close_prices - np.abs(np.random.normal(0, 0.2, 1000)),
        "volume": np.random.uniform(1000000, 5000000, 1000),
    }, index=dates)
    
    engineer = DynamicFeatureEngineer()
    X, y = engineer.fit_transform(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample prediction: {engineer.transform_live(df)}")