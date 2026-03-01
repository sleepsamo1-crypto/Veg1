# ml/db_loader.py
"""Load price data via Django ORM.

Thesis mode (recommended):
- Aggregate ALL markets into one daily series per vegetable.

For each (vegetable_name, date):
  - avg_price = mean across markets
  - min_price = min across markets
  - max_price = max across markets

Why this matters:
- If you train per (veg, market), each pair often has too few days,
  so scripts print "rows too small" and skip.
- Aggregating markets gives you a much longer time series per vegetable.

NOTE: This file assumes Django has been initialized (DJANGO_SETTINGS_MODULE
set + django.setup() called) by the caller script.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from api.models import VegetablePrice


def load_price_data(
    veg: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregate: str = "per_veg",
) -> pd.DataFrame:
    """Load price data.

    Args:
        veg: if provided, filter by vegetable_name
        start_date/end_date: 'YYYY-MM-DD' filters (optional)
        aggregate:
          - 'raw': keep market_name rows
          - 'per_veg': merge all markets into one series per vegetable per day

    Returns:
        DataFrame.
    """

    qs = VegetablePrice.objects.all()

    if veg:
        qs = qs.filter(vegetable_name=veg)
    if start_date:
        qs = qs.filter(date__gte=start_date)
    if end_date:
        qs = qs.filter(date__lte=end_date)

    # Keep these fields for both modes
    df = pd.DataFrame.from_records(
        qs.values(
            "vegetable_name",
            "market_name",
            "min_price",
            "max_price",
            "avg_price",
            "date",
            "crawl_time",
        )
    )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    if aggregate == "raw":
        return (
            df.sort_values(["vegetable_name", "market_name", "date"])
            .reset_index(drop=True)
        )

    if aggregate == "per_veg":
        out = (
            df.groupby(["vegetable_name", "date"], as_index=False)
            .agg(
                {
                    "avg_price": "mean",
                    "min_price": "min",
                    "max_price": "max",
                }
            )
            .sort_values(["vegetable_name", "date"])
            .reset_index(drop=True)
        )
        return out

    raise ValueError(f"Unknown aggregate mode: {aggregate}")


