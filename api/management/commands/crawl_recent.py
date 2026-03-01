import random
import time
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from io import StringIO

import pandas as pd
import requests
from django.core.management.base import BaseCommand
from django.db.models import Max
from django.utils.timezone import now

from api.models import VegetablePrice

# ========== 固定福建 ==========
PROVINCE_CODE = "fujian"
PROVINCE_NAME = "福建"

# 你爬虫里用的品种子域名列表（可按需增减）
TYPE_LIST = ["baicai", "qincai", "bocai", "woju", "hg", "tudou", "fanqie", "qiezi"]

# 网站最早可用日期（你已确认 2026-01-01）
SITE_MIN_DATE = date(2026, 1, 1)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": f"https://baicai.cnveg.com/price/{PROVINCE_CODE}/",
}


def _to_decimal(x) -> Decimal | None:
    """把各种价格格式安全转成 Decimal(保留两位)"""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "--", "None", "nan", "NaN"):
        return None
    s = s.replace("￥", "").replace(",", "")
    try:
        d = Decimal(s)
        return d.quantize(Decimal("0.00"))
    except (InvalidOperation, ValueError):
        return None


def build_url(v_type: str, page: int) -> str:
    # 对齐你旧爬虫的 URL 规则
    return f"http://{v_type}.cnveg.com/price/{PROVINCE_CODE}/p{page}.html"


def _decode_html(resp: requests.Response) -> str:
    """对齐旧代码 utf-8 decode + 中文站兜底"""
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("gb18030", errors="ignore")


def _read_price_table(html_text: str) -> pd.DataFrame:
    """
    对齐你旧逻辑：必须 len(tables) > 16 才取 tables[16]
    """
    tables = pd.read_html(StringIO(html_text))
    if len(tables) <= 16:
        return pd.DataFrame()

    df = pd.DataFrame(tables[16]).copy()
    df.columns = ["vegetable_name", "market_name", "min_price", "max_price", "avg_price", "date"]
    return df


def fetch_page_df(v_type: str, page: int, session: requests.Session, debug: bool = False) -> pd.DataFrame:
    url = build_url(v_type, page)
    resp = session.get(url, headers=HEADERS, timeout=15)

    if debug:
        print(f"  GET {url} -> {resp.status_code}, final_url={resp.url}, len={len(resp.content)}")

    if resp.status_code != 200:
        return pd.DataFrame()

    html_text = _decode_html(resp)
    df = _read_price_table(html_text)

    # 记录“最终URL”，用于判断是否进入最后页循环
    df.attrs["requested_url"] = url
    df.attrs["final_url"] = resp.url

    if df.empty:
        return df

    # 清洗转换（尽量不误杀）
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["min_price"] = df["min_price"].apply(_to_decimal)
    df["max_price"] = df["max_price"].apply(_to_decimal)
    df["avg_price"] = df["avg_price"].apply(_to_decimal)

    df = df.dropna(subset=["date"]).copy()

    # 补齐字段（对齐你的 MySQL）
    df["province_code"] = PROVINCE_CODE
    df["province_name"] = PROVINCE_NAME
    df["category"] = v_type
    df["crawl_time"] = now()

    return df


class Command(BaseCommand):
    help = "准实时增量爬取福建蔬菜价格（最近N天+回补+去重入库+重复页自动停止）。"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7, help="抓取范围天数（默认7天）")
        parser.add_argument("--overlap", type=int, default=7, help="回补重叠天数（默认7天）")
        parser.add_argument("--max-pages", type=int, default=200, help="每个品种最多翻页数（默认200）")
        parser.add_argument("--debug", action="store_true", help="打印调试信息")

    def handle(self, *args, **options):
        days = max(1, int(options["days"]))
        overlap = max(1, int(options["overlap"]))
        max_pages = max(1, int(options["max_pages"]))
        debug = bool(options.get("debug", False))

        agg = VegetablePrice.objects.aggregate(max_date=Max("date"))
        db_max_date = agg["max_date"]

        end_date = date.today()

        # --days 控制范围（最近 days 天）
        recent_start = end_date - timedelta(days=days - 1)
        if recent_start < SITE_MIN_DATE:
            recent_start = SITE_MIN_DATE

        # overlap 回补：再往前扩 overlap-1 天
        start_date = recent_start - timedelta(days=overlap - 1)
        if start_date < SITE_MIN_DATE:
            start_date = SITE_MIN_DATE

        # 如果库里已经到 2026，保证不比（db_max_date - overlap + 1）更晚，避免漏回补
        if db_max_date and db_max_date >= SITE_MIN_DATE:
            db_based = db_max_date - timedelta(days=overlap - 1)
            if db_based < SITE_MIN_DATE:
                db_based = SITE_MIN_DATE
            if db_based < start_date:
                start_date = db_based

        self.stdout.write(
            f"[crawl_recent] db_max_date={db_max_date} start_date={start_date} end_date={end_date} "
            f"(days={days}, overlap={overlap})"
        )

        # 预加载已有 keys（start_date 之后）
        existing = set(
            VegetablePrice.objects.filter(
                province_code=PROVINCE_CODE,
                date__gte=start_date,
                date__lte=end_date,
            ).values_list("vegetable_name", "market_name", "date")
        )

        session = requests.Session()

        inserted = 0
        skipped = 0
        total_seen = 0
        to_create: list[VegetablePrice] = []

        for v_type in TYPE_LIST:
            self.stdout.write(f"\n==> 品种子域名 {v_type}")

            # ✅ 重复页检测：避免最后页循环
            last_final_url = None
            last_signature = None
            repeat_streak = 0

            for page in range(1, max_pages + 1):
                time.sleep(random.uniform(0.3, 0.9))

                raw_df = fetch_page_df(v_type, page, session, debug=debug)
                if raw_df.empty:
                    self.stdout.write(f"  p{page}: empty/invalid, stop.")
                    break

                requested_url = raw_df.attrs.get("requested_url")
                final_url = raw_df.attrs.get("final_url")

                # 这页原始日期范围
                dates = raw_df["date"].tolist()
                page_min = min(dates)
                page_max = max(dates)

                # 过滤到我们关心的范围
                df = raw_df[(raw_df["date"] >= start_date) & (raw_df["date"] <= end_date)]

                if df.empty:
                    if page_min < start_date:
                        self.stdout.write(f"  p{page}: all older than start_date, stop.")
                        break
                    else:
                        self.stdout.write(f"  p{page}: filtered empty but page_min={page_min}, continue.")
                        continue

                # ✅ 构建“页面签名”：日期范围 + 行数 + 唯一日期集合（用于判断是否重复页）
                uniq_dates = tuple(sorted(set(df["date"].tolist())))
                signature = (min(uniq_dates), max(uniq_dates), len(df), uniq_dates)

                # ✅ 只要 final_url 连续相同，或 signature 连续相同，就认为进入“最后页循环”
                if (final_url is not None and final_url == last_final_url) or (signature == last_signature):
                    repeat_streak += 1
                else:
                    repeat_streak = 0

                last_final_url = final_url
                last_signature = signature

                if repeat_streak >= 1:
                    self.stdout.write(
                        f"  p{page}: repeated page detected (requested={requested_url}, final={final_url}), stop."
                    )
                    break

                # 入库去重（幂等）
                for r in df.to_dict(orient="records"):
                    total_seen += 1
                    key = (r["vegetable_name"], r["market_name"], r["date"])
                    if key in existing:
                        skipped += 1
                        continue
                    existing.add(key)
                    to_create.append(VegetablePrice(**r))

                # 分批写入
                if len(to_create) >= 1000:
                    VegetablePrice.objects.bulk_create(to_create, batch_size=1000)
                    inserted += len(to_create)
                    to_create.clear()

                self.stdout.write(
                    f"  p{page}: keep={len(df)} page_min={page_min} page_max={page_max}"
                )

                # 正常停止：已经翻到早于 start_date 的页了
                if page_min < start_date:
                    self.stdout.write(f"  p{page}: reached older than start_date, stop.")
                    break

        # 收尾 flush
        if to_create:
            VegetablePrice.objects.bulk_create(to_create, batch_size=1000)
            inserted += len(to_create)
            to_create.clear()

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. inserted={inserted}, skipped={skipped}, total_seen(after_filter)={total_seen}"
            )
        )
