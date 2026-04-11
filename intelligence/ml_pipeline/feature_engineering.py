from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("FeatureEngineer")

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class DynamicFeatureEngineer:
    """
    Institutional feature generation and dimensionality reduction pipeline.

    Upgrades:
    - Patched pv_trend to use Relative Volume (RVOL) to ensure statistical stationarity
      and remove the intraday U-shape volume bias from the ML model.
    """

    REQUIRED_COLUMNS = ("close", "high", "low", "volume")

    def __init__(
        self,
        variance_retained: float = 0.95,
        min_rows: int = 60,
        target_horizon: int = 5,
    ) -> None:
        if not (0 < variance_retained <= 1):
            raise ValueError("variance_retained must be in the interval (0, 1].")
        if min_rows < 30:
            raise ValueError("min_rows must be at least 30.")
        if target_horizon < 1:
            raise ValueError("target_horizon must be >= 1.")

        self.variance_retained = float(variance_retained)
        self.min_rows = int(min_rows)
        self.target_horizon = int(target_horizon)

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.variance_retained, svd_solver="full")

        self.is_fitted: bool = False
        self.feature_columns: List[str] = []

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < self.min_rows:
            raise ValueError(f"Need at least {self.min_rows} rows. Got {len(df)}.")

    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw, vectorized technical and statistical features.
        All calculations are strictly stationary.
        """
        self._validate_input_df(df)

        work = df.loc[:, ["close", "high", "low", "volume"]].copy()
        work = work.replace([np.inf, -np.inf], np.nan).dropna()

        work = work[
            (work["close"] > 0)
            & (work["high"] > 0)
            & (work["low"] > 0)
            & (work["volume"] >= 0)
            & (work["high"] >= work["low"])
        ]

        if len(work) < self.min_rows:
            raise ValueError(f"Not enough valid rows after cleaning. Need at least {self.min_rows}.")

        features = pd.DataFrame(index=work.index)

        close = work["close"]
        high = work["high"]
        low = work["low"]
        volume = work["volume"]

        # 1. Returns / price velocity
        features["ret_1"] = close.pct_change(1)
        features["ret_5"] = close.pct_change(5)
        features["ret_15"] = close.pct_change(15)
        features["log_ret"] = np.log(close / close.shift(1))

        # 2. Volatility
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        features["natr"] = atr / close.replace(0, np.nan)

        rolling_mean = close.rolling(20).mean()
        rolling_std = close.rolling(20).std()
        features["bb_zscore"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        # 3. Momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # 4. Volume / microstructure dynamics (Stationary Upgrades)
        features["vol_roc_5"] = volume.pct_change(5)
        
        # Calculate Relative Volume (RVOL) to remove time-of-day bias
        rolling_vol_mean = volume.rolling(20).mean().replace(0, np.nan)
        rvol_20 = volume / rolling_vol_mean
        
        # Use RVOL instead of raw volume for the trend multiplier
        features["pv_trend"] = (close.pct_change(1) * rvol_20).rolling(10).sum()
        
        # 5. Extra stability features
        features["hl_spread_pct"] = (high - low) / close.replace(0, np.nan)
        features["close_vs_rolling_mean_10"] = (close / close.rolling(10).mean()) - 1.0
        features["volume_zscore_20"] = (volume - rolling_vol_mean) / volume.rolling(20).std().replace(0, np.nan)

        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        if features.empty:
            raise ValueError("Feature generation resulted in an empty feature set.")

        return features

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Historical training mode:
        - generate features
        - generate future-return target
        - fit scaler
        - fit PCA
        """
        logger.info("Starting historical feature engineering and PCA fitting...")

        feature_df = self._generate_technical_features(df)

        future_returns = df["close"].shift(-self.target_horizon) / df["close"] - 1.0
        combined = pd.concat([feature_df, future_returns.rename("target")], axis=1)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

        if combined.empty:
            raise ValueError("No usable rows remain after aligning features and target.")

        X_raw = combined.drop(columns=["target"])
        y_target = combined["target"]

        self.feature_columns = X_raw.columns.tolist()

        X_scaled = self.scaler.fit_transform(X_raw)
        X_pca = self.pca.fit_transform(X_scaled)

        self.is_fitted = True

        logger.info(
            "Feature Space Compressed: %d raw features -> %d principal components.",
            len(self.feature_columns),
            self.pca.n_components_,
        )

        return X_pca, y_target

    def transform_live(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Live execution mode:
        transform the latest available bar using the fitted scaler + PCA state.
        """
        if not self.is_fitted:
            logger.error("Attempted to transform live data before fitting the feature engineer.")
            return None

        try:
            feature_df = self._generate_technical_features(df)
        except ValueError as e:
            logger.warning("Live feature generation skipped: %s", e)
            return None

        if feature_df.empty:
            return None

        latest_features = feature_df.iloc[-1:].copy()

        missing_cols = set(self.feature_columns) - set(latest_features.columns)
        if missing_cols:
            logger.error("Live data is missing fitted feature columns: %s", sorted(missing_cols))
            return None

        latest_features = latest_features[self.feature_columns]
        latest_features = latest_features.replace([np.inf, -np.inf], np.nan).dropna()
        
        if latest_features.empty:
            return None

        X_scaled_live = self.scaler.transform(latest_features)
        X_pca_live = self.pca.transform(X_scaled_live)

        return X_pca_live


if __name__ == "__main__":
    print("Initializing Dynamic Feature Engineer Test...")

    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=1000, freq="5min")
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.002, 1000)))

    df_historical = pd.DataFrame(
        {
            "open": close_prices * (1 + np.random.normal(0, 0.001, 1000)),
            "high": close_prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
            "low": close_prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
            "close": close_prices,
            "volume": np.random.randint(1000, 50000, 1000),
        },
        index=dates,
    )

    engineer = DynamicFeatureEngineer(variance_retained=0.95)

    print("\n[HISTORICAL] Fitting PCA and Scaler...")
    X_train, y_train = engineer.fit_transform(df_historical)
    print(f"Historical Output Shape: {X_train.shape} (Rows, Principal Components)")
    print(f"Target Shape: {y_train.shape}")

    print("\n[LIVE INFERENCE] Processing latest market bar...")
    df_live = df_historical.copy()
    df_live.loc[df_live.index[-1] + pd.Timedelta(minutes=5)] = [105.0, 105.5, 104.8, 105.2, 25000]

    X_live = engineer.transform_live(df_live)
    if X_live is not None:
        print(f"Live Vector Shape for ML Model: {X_live.shape}")
        print(f"Live Compressed Feature Vector: {np.round(X_live[0], 3)}")
    else:
        print("Live transform returned None.")