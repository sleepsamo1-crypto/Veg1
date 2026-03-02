# ml/features.py
"""Feature engineering for vegetable price time series.

Design goals (good for a thesis writeup):
- No leakage: features at day t only depend on history up to t-1.
- Interpretable: lag + rolling statistics + calendar features.
- Extensible: you can add more (holiday, weather) later, but not required.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def build_time_series_features_grouped(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("vegetable_name",),
    target_col: str = "avg_price",
    date_col: str = "date",
    lags: Sequence[int] = (1, 2, 3, 7, 14),
    windows: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    """Build features for grouped time series (NO leakage).

    Notes
    -----
    - Lag features: y_{t-k}
    - Rolling features: computed per-group on shifted series (history only)
    - Diff features: computed from lagged values (e.g. diff_1 = lag_1 - lag_2)
      so they do not use y_t.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(list(group_cols) + [date_col]).reset_index(drop=True)

    g = out.groupby(list(group_cols), sort=False)

    # --- Calendar features (safe, known in advance) ---
    out["dow"] = out[date_col].dt.weekday  # 0=Mon
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["quarter"] = out[date_col].dt.quarter
    out["day_of_year"] = out[date_col].dt.dayofyear

    doy = out["day_of_year"].astype(float)
    out["sin_doy_1y"] = np.sin(2 * np.pi * doy / 365.0)
    out["cos_doy_1y"] = np.cos(2 * np.pi * doy / 365.0)
    # --- Lag features ---
    for k in lags:
        out[f"lag_{k}"] = g[target_col].shift(k)

    # --- Rolling features (shift(1) first to avoid leakage; computed per group) ---
    def _roll(rolling_fn, w: int) -> pd.Series:
        # result index: MultiIndex (group_keys + original index) -> drop group levels
        return (
            g[target_col]
            .apply(lambda s: rolling_fn(s.shift(1).rolling(w)))
            .reset_index(level=list(group_cols), drop=True)
        )

    for w in windows:
        out[f"roll_mean_{w}"] = _roll(lambda r: r.mean(), w)
        out[f"roll_std_{w}"] = _roll(lambda r: r.std(), w)   # pandas default ddof=1
        out[f"roll_min_{w}"] = _roll(lambda r: r.min(), w)
        out[f"roll_max_{w}"] = _roll(lambda r: r.max(), w)

    # --- Differences (use lagged values only; NO leakage) ---
    # diff_1 at day t = y_{t-1} - y_{t-2}
    out["diff_1"] = out["lag_1"] - out["lag_2"]

    for k in (7, 14):
        if (k in lags) or (k in windows):
            # diff_k at day t = y_{t-1} - y_{t-(k+1)}
            lk1 = f"lag_{k+1}"
            if lk1 not in out.columns:
                out[lk1] = g[target_col].shift(k + 1)
            out[f"diff_{k}"] = out["lag_1"] - out[lk1]

    # clean infinities
    for c in out.columns:
        if c.startswith("roll_"):
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return model feature column names."""
    base = {
        "date",
        "avg_price",
        "min_price",
        "max_price",
        "market_name",
        "vegetable_name",
        "crawl_time",
    }
    cols = [c for c in df.columns if c not in base]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def build_features(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("vegetable_name",),
    target_col: str = "avg_price",
    date_col: str = "date",
    lags: Sequence[int] = (1, 2, 3, 7, 14),
    windows: Sequence[int] = (7, 14),
) -> tuple[pd.DataFrame, list[str]]:
    """Convenience wrapper used by scripts."""
    df_feat = build_time_series_features_grouped(
        df,
        group_cols=group_cols,
        target_col=target_col,
        date_col=date_col,
        lags=lags,
        windows=windows,
    )
    feature_cols = get_feature_cols(df_feat)
    return df_feat, feature_cols
