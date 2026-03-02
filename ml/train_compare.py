# ml/train_compare.py (重构版 - 模块化)
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.django_bootstrap import bootstrap_django

settings_mod = bootstrap_django()
print("[Django] settings:", settings_mod)

from ml.db_loader import load_price_data  # noqa: E402
from ml.features import build_features  # noqa: E402
from ml.metrics import compute_all_metrics  # noqa: E402
from ml.models.sklearn_models import build_sklearn_models  # noqa: E402
from ml.models.xgboost_model import build_xgboost_model, XGBOOST_AVAILABLE  # noqa: E402
from ml.models.lstm_model import build_lstm_model, LSTM_AVAILABLE  # noqa: E402
from ml.tuning.hyperparameters import get_hyperparameters  # noqa: E402


def train_top3_for_veg(
    veg: str,
    horizon: int = 7,
    min_rows: int = 120,
    lags=(1, 2, 3, 7, 14),
    windows=(7, 14),
    decimals: int = 2,
    out_dir: str = "saved_models/by_veg",
    hp_mode: str = "default",  # "default" | "aggressive" | "fast"
    enable_xgboost: bool = True,
    enable_lstm: bool = False,
) -> dict:
    """训练Top-3模型
    
    Args:
        hp_mode: 超参数配置模式
                - "default": 平衡（推荐）
                - "aggressive": 追求精度
                - "fast": 快速训练
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 训练: {veg} (模式: {hp_mode})")
    print(f"{'='*70}")
    
    # 加载超参数
    hp_config = get_hyperparameters(hp_mode)
    
    # 数据加载和特征工程
    df = load_price_data(veg=veg, aggregate="per_veg")
    if df.empty:
        raise ValueError(f"No data found for veg={veg}")
    
    df_feat, feature_cols = build_features(
        df,
        group_cols=("vegetable_name",),
        lags=lags,
        windows=windows,
    )
    
    df_feat = df_feat.dropna(subset=feature_cols + ["avg_price"]).reset_index(drop=True)
    if len(df_feat) < min_rows:
        raise ValueError(f"rows too small: {len(df_feat)} (<{min_rows})")
    
    X = df_feat[feature_cols].values
    y = df_feat["avg_price"].astype(float).values
    
    # 时间序列分割
    test_size = max(horizon, int(len(df_feat) * 0.2))
    test_size = min(test_size, max(10, len(df_feat) - 10))
    split = len(df_feat) - test_size
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"✓ 数据: {len(X_train)} 训练 + {len(X_test)} 测试")
    print(f"✓ 特征: {len(feature_cols)} 维度")
    
    # 构建模型
    models = {}
    models.update(build_sklearn_models(hp_config))
    
    if enable_xgboost and XGBOOST_AVAILABLE:
        try:
            models["XGBoost"] = build_xgboost_model(hp_config)
        except Exception as e:
            print(f"[WARNING] XGBoost 加载失败: {e}")
    
    results = []
    fitted = {}
    
    # 训练模型
    print(f"\n📊 训练 {len(models)} 个模型...")
    for name, model in models.items():
        try:
            print(f"  • {name}...", end=" ", flush=True)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            metrics = {"model": name}
            metrics.update(compute_all_metrics(y_test, pred, y_train, period=7))
            
            results.append(metrics)
            fitted[name] = model.get_model_object()
            
            print(f"✓ RMSE={metrics['RMSE']:.4f}")
        except Exception as e:
            print(f"✗ {e}")
    
    # 排行榜和保存
    leaderboard = pd.DataFrame(results).sort_values("RMSE", ascending=True)
    top3 = leaderboard.head(3).to_dict(orient="records")
    
    print(f"\n🏆 排行榜:")
    print(leaderboard.head(5).to_string(index=False))
    
    veg_dir = Path(out_dir) / veg
    (veg_dir / "models").mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    leaderboard.to_csv(veg_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top3).to_csv(veg_dir / "top3.csv", index=False, encoding="utf-8-sig")
    
    pack = {
        "veg": veg,
        "horizon": horizon,
        "group_mode": "per_veg",
        "decimals": int(decimals),
        "feature_cols": feature_cols,
        "lags": list(lags),
        "windows": list(windows),
        "top3": top3,
        "hp_mode": hp_mode,  # 记录使用的参数模式
    }
    
    with open(veg_dir / "top3.json", "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    
    # 保存Top3模型
    for row in top3:
        name = row["model"]
        joblib.dump(fitted[name], veg_dir / "models" / f"{name}.joblib")
    
    print(f"\n✅ 完成! 输出: {veg_dir}")
    return pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veg", required=True)
    ap.add_argument("--hp-mode", default="default", 
                   choices=["default", "aggressive", "fast"],
                   help="超参数模式")
    ap.add_argument("--enable-xgboost", action="store_true", default=True)
    ap.add_argument("--disable-xgboost", action="store_true")
    ap.add_argument("--enable-lstm", action="store_true")
    ap.add_argument("--out-dir", default="saved_models/by_veg")
    
    args = ap.parse_args()
    
    train_top3_for_veg(
        veg=args.veg,
        hp_mode=args.hp_mode,
        enable_xgboost=args.enable_xgboost and not args.disable_xgboost,
        enable_lstm=args.enable_lstm,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()