# ml/train_entry.py (enhanced features)
import os, sys, django
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\\Veg
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from ml.db_loader import load_price_data
from ml import features  # 把 features_ENHANCED.py 覆盖到你的 ml/features.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    df = load_price_data()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["min_price", "max_price", "avg_price"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    group_cols = ["vegetable_name", "market_name"]
    df_feat = features.build_time_series_features_grouped(
        df,
        group_cols=group_cols,
        target_col="avg_price",
        lags=(1,2,3,7,14),
        windows=(7,14)
    )
    feature_cols = features.get_feature_cols(df_feat)

    df_train = df_feat.dropna(subset=feature_cols + ["avg_price"]).sort_values("date").copy()
    split_idx = int(len(df_train) * 0.8)

    X = df_train[feature_cols].values
    y = df_train["avg_price"].values
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, pred) ** 0.5
    print("Validation RMSE:", rmse)

    out_dir = Path(BASE_DIR) / "saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = {
        "model": model,
        "feature_cols": feature_cols,
        "group_cols": group_cols,
        "target_col": "avg_price",
        "lags": (1,2,3,7,14),
        "windows": (7,14),
        "use_spread": "spread" in feature_cols,
    }
    joblib.dump(pack, out_dir / "rf_avg_price_enhanced.joblib")
    print("Saved:", out_dir / "rf_avg_price_enhanced.joblib")

