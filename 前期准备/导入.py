import pandas as pd
from api.models import VegetablePrice


df = pd.read_csv(r"D:\pythonProject1\data\fujian_clean.csv")

for _, row in df.iterrows():
    VegetablePrice.objects.create(
        vegetable_name=row["vegetable_name"],
        market_name=row["market_name"],
        min_price=row["min_price"],
        max_price=row["max_price"],
        avg_price=row["avg_price"],
        date=row["date"],
        crawl_time=row["timestamp"],
        province_code=row["province_code"],
        province_name=row["province_name"],
        category=row["category"]
    )
