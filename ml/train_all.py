# ml/train_all.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ml.django_bootstrap import bootstrap_django

settings_mod = bootstrap_django()
print("[Django] settings:", settings_mod)

from ml.db_loader import load_price_data  # noqa: E402
from ml.train_compare import train_top3_for_veg  # noqa: E402


def main() -> None:
    base_out = Path("saved_models") / "by_veg"
    base_out.mkdir(parents=True, exist_ok=True)
    print("Output dir:", base_out.resolve())
    
    df_all = load_price_data(aggregate="per_veg")
    if df_all.empty:
        print("No data in DB.")
        return
    
    vegs = sorted(df_all["vegetable_name"].unique().tolist())
    summary = []
    
    # 选择超参数模式
    HP_MODE = "default"  # 修改为 "aggressive" 或 "fast"
    
    for veg in vegs:
        try:
            pack = train_top3_for_veg(
                veg=veg,
                hp_mode=HP_MODE,
                enable_xgboost=True,
                enable_lstm=False,
                out_dir=str(base_out),
            )
            best = pack["top3"][0]
            summary.append({
                "vegetable_name": veg,
                "best_model": best["model"],
                "RMSE": best["RMSE"],
                "HP_Mode": HP_MODE,
            })
        except Exception as e:
            print(f"[SKIP] {veg}: {e}")
    
    if summary:
        pd.DataFrame(summary).to_csv(
            base_out / "summary.csv",
            index=False,
            encoding="utf-8-sig"
        )
        print("\n 完成!")
        print(pd.DataFrame(summary))


if __name__ == "__main__":
    main()