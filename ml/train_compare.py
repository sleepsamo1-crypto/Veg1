# ml/train_compare.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- Django init (robust) ----
from ml.django_bootstrap import bootstrap_django  # noqa: E402

settings_mod = bootstrap_django()
print("[Django] settings:", settings_mod)


from ml.db_loader import load_price_data  # noqa: E402
from ml.features import build_features  # noqa: E402


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def train_top3_for_veg(
    veg: str,
    horizon: int = 7,
    min_rows: int = 120,
    lags=(1, 2, 3, 7, 14),
    windows=(7, 14),
    decimals: int = 2,
    out_dir: str = "saved_models/by_veg",
) -> dict:
    """Train and save top-3 models for ONE vegetable (markets merged -> one series)."""

    df = load_price_data(veg=veg, aggregate="per_veg")
    if df.empty:
        raise ValueError(f"No data found for veg={veg}")

    df_feat, feature_cols = build_features(
        df,
        group_cols=("vegetable_name",),
        lags=lags,
        windows=windows,
    )

    # Drop rows where any feature is missing (caused by lags/rolling at the beginning)
    df_feat = df_feat.dropna(subset=feature_cols + ["avg_price"]).reset_index(drop=True)
    if len(df_feat) < min_rows:
        raise ValueError(f"rows too small after feature engineering: {len(df_feat)} (<{min_rows})")

    X = df_feat[feature_cols]
    y = df_feat["avg_price"].astype(float).values

    # time-based split: last `test_size` rows as test
    test_size = max(horizon, int(len(df_feat) * 0.2))
    test_size = min(test_size, max(10, len(df_feat) - 10))
    split = len(df_feat) - test_size

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),  # ✅ Ridge 没有 random_state
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "model": name,
            "RMSE": rmse(y_test, pred),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "MAPE(%)": mape(y_test, pred),
        }
        results.append(metrics)
        fitted[name] = model
        print(f"[{name}] RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE(%)']:.2f}%")

    leaderboard = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    top3 = leaderboard.head(3).to_dict(orient="records")

    veg_dir = Path(out_dir) / veg
    (veg_dir / "models").mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(veg_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top3).to_csv(veg_dir / "top3.csv", index=False, encoding="utf-8-sig")

    pack = {
        "veg": veg,
        "horizon": horizon,
        "group_mode": "per_veg",
        "target_col": "avg_price",
        "date_col": "date",
        "decimals": int(decimals),
        "feature_cols": feature_cols,
        "lags": list(lags),
        "windows": list(windows),
        "top3": top3,
    }

    # ✅ 统一“真源”：top3.json
    with open(veg_dir / "top3.json", "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    # ✅ 兼容：额外写一个 config.json（旧预测代码读它时不会炸）
    config = {
        "feature_cols": feature_cols,
        "lags": list(lags),
        "windows": list(windows),
        "decimals": int(decimals),
    }
    with open(veg_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for row in top3:
        name = row["model"]
        joblib.dump(fitted[name], veg_dir / "models" / f"{name}.joblib")

    best_name = top3[0]["model"]
    joblib.dump(fitted[best_name], veg_dir / "best_model.joblib")

    print(f"\nSaved: {veg_dir.resolve()}")
    return pack


# ---- compatibility: some scripts previously used this name ----
def train_one_vegetable(veg: str, horizon: int = 7, decimals: int = 2):
    return train_top3_for_veg(veg=veg, horizon=horizon, decimals=decimals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veg", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--out_dir", default="saved_models/by_veg")
    args = ap.parse_args()

    train_top3_for_veg(args.veg, horizon=args.horizon, decimals=args.decimals, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
