from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)


class RegimeClassifierHMM:
    """
    Hidden Markov Model classifier for latent market regime detection.

    Regime labels are assigned by state profile:
    - lower volatility state(s)
    - middle / directional state
    - high volatility / shock state

    Designed for:
    - offline fit on historical OHLC data
    - live latest-regime inference after fit
    
    Usage:
        classifier = RegimeClassifierHMM(n_components=3)
        df_with_regimes, regime_map = classifier.fit_predict(historical_ohlc_df)
        current_regime = classifier.predict_latest_regime(df_latest)
    """

    REQUIRED_COLUMNS = ("close", "high", "low")

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = 42,
        n_iter: int = 1000,
        tol: float = 1e-4,
        min_rows: int = 50,
    ) -> None:
        if n_components < 2:
            raise ValueError("n_components must be >= 2")

        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_rows = int(min_rows)

        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=self.tol,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_mapping: Dict[int, str] = {}

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns and sufficient rows."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame must contain columns: {self.REQUIRED_COLUMNS}. Missing: {missing}")

        if len(df) < 3:
            raise ValueError("DataFrame must contain at least 3 rows.")

    def _feature_engineering(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract stationary features for HMM convergence:
        1. Log returns (stationary price change)
        2. Parkinson volatility (intrabar range-based volatility)
        
        Returns:
            np.ndarray: (n_samples, 2) matrix of [log_returns, volatility]
        """
        self._validate_input_df(df)

        work = df.loc[:, ["close", "high", "low"]].copy()

        # Basic cleaning: remove inf, nan, negative prices
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        work = work[(work["close"] > 0) & (work["high"] > 0) & (work["low"] > 0)]
        work = work[work["high"] >= work["low"]]

        if len(work) < 3:
            raise ValueError("Not enough valid rows after cleaning OHLC data.")

        # Log returns: ln(close_t / close_t-1)
        log_returns = np.log(work["close"] / work["close"].shift(1))
        
        # Parkinson volatility: sqrt( (1/(4*ln(2))) * ln(high/low)^2 )
        parkinson_vol = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * (np.log(work["high"] / work["low"]) ** 2)
        )

        features = pd.DataFrame(
            {
                "log_returns": log_returns,
                "volatility": parkinson_vol,
            },
            index=work.index,
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(features) < max(2, self.n_components * 5):
            raise ValueError(
                f"Insufficient usable feature rows for HMM fit/predict. "
                f"Have {len(features)}, need at least {max(2, self.n_components * 5)}."
            )

        return features.values

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Fit HMM on historical data, predict latent states,
        and map the states to semantic regime labels.
        
        Args:
            df: DataFrame with columns [close, high, low]
        
        Returns:
            (df_with_regimes, regime_mapping_dict)
        """
        if len(df) < self.min_rows:
            raise ValueError(f"Need at least {self.min_rows} rows to fit HMM robustly.")

        feature_matrix = self._feature_engineering(df)
        scaled_features = self.scaler.fit_transform(feature_matrix)

        # Fit HMM to scaled features
        self.model.fit(scaled_features)
        hidden_states = self.model.predict(scaled_features)
        self.is_fitted = True

        # Characterize each state by mean volatility and mean return
        state_profiles = []
        for state_id in range(self.n_components):
            state_data = feature_matrix[hidden_states == state_id]
            if len(state_data) == 0:
                state_profiles.append((state_id, float("inf"), 0.0))
                continue

            mean_vol = float(np.mean(state_data[:, 1]))
            mean_return = float(np.mean(state_data[:, 0]))
            state_profiles.append((state_id, mean_vol, mean_return))

        # Sort by volatility (ascending)
        state_profiles.sort(key=lambda x: x[1])

        # Assign semantic labels based on volatility profile
        if self.n_components == 3:
            self.regime_mapping = {
                state_profiles[0][0]: "Low Volatility / Range Bound",
                state_profiles[1][0]: "Directional Trend",
                state_profiles[2][0]: "High Volatility / Shock",
            }
        else:
            self.regime_mapping = {
                state[0]: f"Regime_{idx}"
                for idx, state in enumerate(state_profiles)
            }

        df_result = df.copy()

        # feature engineering loses the first row due to shift(1)
        padded_states = np.full(len(df_result), -1, dtype=int)
        padded_states[-len(hidden_states):] = hidden_states

        df_result["Regime_ID"] = padded_states
        df_result["Regime_Label"] = (
            df_result["Regime_ID"].map(self.regime_mapping).fillna("Insufficient Data")
        )

        return df_result, self.regime_mapping

    def predict_latest_regime(self, df: pd.DataFrame) -> Optional[str]:
        """
        Predict the current regime using the latest available rows.
        
        Args:
            df: DataFrame with columns [close, high, low]
        
        Returns:
            Regime label string, or None if prediction fails
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted via fit_predict() before predict_latest_regime().")

        if df is None or len(df) < 3:
            return None

        # Use a small recent window (20 rows) for stable inference
        recent = df.tail(20).copy()

        try:
            feature_matrix = self._feature_engineering(recent)
        except ValueError:
            return None

        if len(feature_matrix) == 0:
            return None

        scaled_features = self.scaler.transform(feature_matrix)
        latest_state = int(self.model.predict(scaled_features)[-1])

        return self.regime_mapping.get(latest_state, "Unknown State")


if __name__ == "__main__":
    print("Initializing HMM Regime Classifier Test...")

    # Generate synthetic data with regime changes
    dates = pd.date_range("2025-01-01", periods=500, freq="h")

    np.random.seed(42)
    quiet_returns = np.random.normal(0, 0.001, 250)
    volatile_returns = np.random.normal(0, 0.005, 250)
    all_returns = np.concatenate([quiet_returns, volatile_returns])

    close_prices = 100 * np.exp(np.cumsum(all_returns))
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, 500)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, 500)))

    dummy_df = pd.DataFrame(
        {"close": close_prices, "high": high_prices, "low": low_prices},
        index=dates,
    )

    classifier = RegimeClassifierHMM(n_components=3)
    analyzed_df, regime_map = classifier.fit_predict(dummy_df)

    print("\nRegime Mapping Discovered:")
    for k, v in regime_map.items():
        print(f"State {k}: {v}")

    print("\nLatest Predicted Regime:", classifier.predict_latest_regime(dummy_df))
    
    print("\nFirst 10 rows with regime labels:")
    print(analyzed_df[["close", "Regime_ID", "Regime_Label"]].head(10))