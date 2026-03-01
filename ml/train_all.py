# ml/train_all.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.django_bootstrap import bootstrap_django

# 1) Django 初始化（自动读取 manage.py 里的 DJANGO_SETTINGS_MODULE）
settings_mod = bootstrap_django()
print("[Django] settings:", settings_mod)

from ml.db_loader import load_price_data  # noqa: E402
from ml.train_compare import train_top3_for_veg  # noqa: E402


def main() -> None:
    base_out = Path("saved_models") / "by_veg"
    base_out.mkdir(parents=True, exist_ok=True)
    print("Output dir:", base_out.resolve())

    # markets 合并 -> per_veg
    df_all = load_price_data(aggregate="per_veg")
    if df_all.empty:
        print("No data in DB. Please crawl/import first.")
        return

    vegs = sorted(df_all["vegetable_name"].unique().tolist())
    summary = []

    for veg in vegs:
        try:
            print(f"\n=== Training: {veg} ===")
            pack = train_top3_for_veg(
                veg=veg,
                horizon=7,
                min_rows=120,
                out_dir=str(base_out),
            )
            best = pack["top3"][0]
            summary.append(
                {
                    "vegetable_name": veg,
                    "best_model": best["model"],
                    "RMSE": best["RMSE"],
                    "MAE": best.get("MAE"),
                    "MAPE(%)": best.get("MAPE(%)"),
                }
            )
        except Exception as e:
            print(f"[SKIP] {veg}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv(base_out / "summary.csv", index=False, encoding="utf-8-sig")
        print("\nSaved summary:", (base_out / "summary.csv").resolve())


if __name__ == "__main__":
    main()
