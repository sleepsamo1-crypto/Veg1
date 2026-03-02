# ml/predict_entry.py
"""Forecast helper used by Django views.

Given:
- a vegetable name
- a model pack saved by ml/train_compare.py (Top-3)

Return:
- historical dates + real prices
- next N days forecast for each of the top-3 models
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def _pack_dir(veg: str) -> Path:
    return Path("saved_models") / "by_veg" / veg


def load_pack(veg: str) -> dict:
    p = _pack_dir(veg) / "top3.json"
    if not p.exists():
        raise FileNotFoundError(
            f"top3.json not found for veg={veg}. Run train_all.py or train_compare.py first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_top3(veg: str) -> List[dict]:
    return load_pack(veg)["top3"]


def _build_next_row_features(
    history_prices: pd.Series,
    next_date: pd.Timestamp,
    feature_cols: List[str],
    lags: List[int],
    windows: List[int],
) -> Dict[str, float]:
    """Create features for `next_date` using only history (NO leakage)."""
    s = history_prices.astype(float)

    row: Dict[str, float] = {}

    # calendar (must match features.py)
    row["dow"] = int(next_date.weekday())
    row["month"] = int(next_date.month)
    row["day"] = int(next_date.day)
    row["is_weekend"] = 1 if row["dow"] >= 5 else 0
    row["quarter"] = int((next_date.month - 1) // 3 + 1)
    row["day_of_year"] = int(next_date.dayofyear)

    doy = float(row["day_of_year"])
    row["sin_doy_1y"] = float(np.sin(2 * np.pi * doy / 365.0))
    row["cos_doy_1y"] = float(np.cos(2 * np.pi * doy / 365.0))

    # lags
    for k in lags:
        row[f"lag_{k}"] = float(s.iloc[-k]) if len(s) >= k else np.nan

    # rolling stats (match features.py: per-group shifted rolling; here history is already "known up to t-1")
    for w in windows:
        tail = s.iloc[-w:] if len(s) >= w else s
        row[f"roll_mean_{w}"] = float(tail.mean()) if len(tail) else np.nan
        row[f"roll_std_{w}"] = float(tail.std(ddof=1)) if len(tail) > 1 else np.nan
        row[f"roll_min_{w}"] = float(tail.min()) if len(tail) else np.nan
        row[f"roll_max_{w}"] = float(tail.max()) if len(tail) else np.nan

    # diffs (must match features.py)
    row["diff_1"] = row.get("lag_1", np.nan) - row.get("lag_2", np.nan)

    for k in (7, 14):
        if (k in lags) or (k in windows):
            lk1 = f"lag_{k+1}"
            if lk1 not in row:
                row[lk1] = float(s.iloc[-(k + 1)]) if len(s) >= (k + 1) else np.nan
            row[f"diff_{k}"] = row.get("lag_1", np.nan) - row.get(lk1, np.nan)

    return {c: row.get(c, np.nan) for c in feature_cols}


def forecast_top3(
    df_hist: pd.DataFrame,
    veg: str,
    horizon: int = 7,
) -> Tuple[List[str], List[float], List[str], Dict[str, List[float]]]:
    """Return historical + future forecasts (Top-3 models) for ONE veg."""
    if df_hist is None or df_hist.empty:
        raise ValueError("df_hist is empty")

    df_hist = df_hist.copy()
    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_hist = df_hist.sort_values("date").reset_index(drop=True)

    pack = load_pack(veg)
    top3 = pack["top3"]
    feature_cols: List[str] = pack["feature_cols"]
    lags: List[int] = pack["lags"]
    windows: List[int] = pack["windows"]
    decimals: int = int(pack.get("decimals", 2))

    # load model objects
    models = {}
    for item in top3:
        name = item["model"]
        path = _pack_dir(veg) / "models" / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}. Please re-train.")
        models[name] = joblib.load(path)

    hist_dates = df_hist["date"].dt.strftime("%Y-%m-%d").tolist()
    hist_prices = df_hist["avg_price"].astype(float).round(decimals).tolist()

    # one evolving history per model
    base_prices = df_hist["avg_price"].astype(float)
    evolving: Dict[str, pd.Series] = {m: base_prices.copy() for m in models.keys()}

    future_dates: List[str] = []
    future_preds_by_model: Dict[str, List[float]] = {k: [] for k in models.keys()}

    last_date = df_hist["date"].iloc[-1]

    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        future_dates.append(next_date.strftime("%Y-%m-%d"))

        for model_name, model_obj in models.items():
            feat_row = _build_next_row_features(
                history_prices=evolving[model_name],
                next_date=next_date,
                feature_cols=feature_cols,
                lags=lags,
                windows=windows,
            )
            X_next = pd.DataFrame([feat_row], columns=feature_cols)
            y_next = float(model_obj.predict(X_next)[0])
            y_next_round = round(y_next, decimals)

            future_preds_by_model[model_name].append(y_next_round)
            evolving[model_name] = pd.concat(
                [evolving[model_name], pd.Series([y_next])],
                ignore_index=True
            )

    return hist_dates, hist_prices, future_dates, future_preds_by_model
