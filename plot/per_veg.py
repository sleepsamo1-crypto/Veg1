import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import math

# ====== 0) 全局配置 ======
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

CSV_PATH = r"D:\Veg\data\fujian_clean.csv"
UNIT = "元/斤"
BOX_VEG = "茄子"
CAL_VEG = "土豆"

# 缺失值显示颜色（热力图）
HEAT_NAN_COLOR = "lightgray"

# ====== 1) 读数据 & 清洗 ======
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df["date"] = pd.to_datetime(df["date"])
df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce")
df["min_price"] = pd.to_numeric(df.get("min_price", np.nan), errors="coerce")
df["max_price"] = pd.to_numeric(df.get("max_price", np.nan), errors="coerce")

# ====== 2) 三市场聚合（按 蔬菜+日期） ======
per_veg = (
    df.groupby(["vegetable_name", "date"], as_index=False)
      .agg(avg_price=("avg_price", "mean"),
           min_price=("min_price", "min"),
           max_price=("max_price", "max"))
)
per_veg["month"] = per_veg["date"].dt.month

# 月份范围（只画有数据的月份，避免空白）
months_have = sorted(per_veg["month"].dropna().unique().tolist())
m_min, m_max = int(min(months_have)), int(max(months_have))

# ====== 3) 月均价（按 蔬菜+年月）用于折线图 ======
per_day = per_veg[["vegetable_name", "date", "avg_price"]].copy()
per_day["month_ts"] = per_day["date"].dt.to_period("M").dt.to_timestamp()

per_month = (
    per_day.groupby(["vegetable_name", "month_ts"], as_index=False)
           .agg(avg_price=("avg_price", "mean"))
           .sort_values(["vegetable_name", "month_ts"])
)

all_vegs = sorted(per_month["vegetable_name"].unique())

en_map = {
    "土豆": "Potato",
    "西红柿": "Tomato",
    "大白菜": "Chinese cabbage",
    "小白菜": "Pak choi",
    "油麦菜": "Leaf lettuce",
    "洋白菜": "Cabbage",
    "生菜": "Lettuce",
    "芹菜": "Celery",
    "茄子": "Eggplant",
    "莴笋": "Lettuce stem",
    "菠菜": "Spinach",
    "黄瓜": "Cucumber",
}

# ====== A) 箱线图（单蔬菜：按月份分布） ======
d_box = per_veg[per_veg["vegetable_name"] == BOX_VEG].copy()
data = [d_box.loc[d_box["month"] == m, "avg_price"].dropna().values for m in range(m_min, m_max + 1)]

plt.figure(figsize=(10, 4))
plt.boxplot(data, tick_labels=[str(m) for m in range(m_min, m_max + 1)], showfliers=False)
plt.xlabel("月份")
plt.ylabel(f"平均价格（{UNIT}）")
plt.title(f"{BOX_VEG}按月份价格分布箱线图")
plt.tight_layout()
plt.show()

# ====== B) 热力图（蔬菜 × 月份：月均价） ======
pivot = (
    per_veg.groupby(["vegetable_name", "month"])["avg_price"]
           .mean()
           .unstack("month")
           .reindex(columns=range(m_min, m_max + 1))
)

# 1) 排序（按全年均价从低到高）
order = pivot.mean(axis=1).sort_values().index
pivot2 = pivot.loc[order]

# 2) 分位数截断（增强对比）
vals = pivot2.values
vmin, vmax = np.nanpercentile(vals, [5, 95])

# NaN 显示灰色
cmap = plt.cm.viridis.copy()
cmap.set_bad(color=HEAT_NAN_COLOR)

plt.figure(figsize=(12, max(5, 0.35 * len(pivot2))))
plt.imshow(pivot2.values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
plt.yticks(range(len(pivot2.index)), pivot2.index)
plt.xticks(range(pivot2.shape[1]), [str(m) for m in range(m_min, m_max + 1)])
plt.xlabel("月份")
plt.ylabel("蔬菜品种")
plt.title("蔬菜×月份月均价热力图")
plt.colorbar(label=f"平均价格（{UNIT}）")
plt.tight_layout()
plt.show()

# ====== C) 日历热力图（月份 × 日：单蔬菜） ======
d_cal = per_veg[per_veg["vegetable_name"] == CAL_VEG].copy()
d_cal["day"] = d_cal["date"].dt.day

cal = (
    d_cal.groupby(["month", "day"])["avg_price"]
         .mean()
         .unstack("day")
         .reindex(index=range(m_min, m_max + 1))  # 只画有数据的月份
)

cmap2 = plt.cm.viridis.copy()
cmap2.set_bad(color=HEAT_NAN_COLOR)

plt.figure(figsize=(14, 4))
plt.imshow(cal.values, aspect="auto", cmap=cmap2)
plt.yticks(range(cal.shape[0]), [str(m) for m in range(m_min, m_max + 1)])
plt.xticks(range(cal.shape[1]), cal.columns.astype(str), rotation=90)
plt.xlabel("日期（日）")
plt.ylabel("月份")
plt.title(f"{CAL_VEG}日历热力图（月份×日）")
plt.colorbar(label=f"平均价格（{UNIT}）")
plt.tight_layout()
plt.show()

# ====== D) 折线图（全部品种：按月均价，带 marker，图例中文+英文） ======
plt.figure(figsize=(14, 6))

markers = ["o","s","^","D","x","*","v","P","<",">","h","H","1","2","3","4"]
for i, veg in enumerate(all_vegs):
    d = per_month[per_month["vegetable_name"] == veg]
    if d.empty:
        continue
    label = f"{veg} {en_map.get(veg, '')}".strip()
    plt.plot(
        d["month_ts"], d["avg_price"],
        marker=markers[i % len(markers)],
        markersize=3,
        linewidth=1.2,
        alpha=0.9,
        label=label
    )

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=90)

plt.xlabel("月份")
plt.ylabel(f"平均价格（{UNIT}）")
plt.title(f"各蔬菜月均价变化折线图")

ncol = max(1, math.ceil(len(all_vegs) / 12))
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    ncol=ncol,
    fontsize=9,
    frameon=False
)

plt.tight_layout()
plt.show()
