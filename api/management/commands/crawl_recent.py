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
from django.conf import settings
import datetime
from django.utils import timezone

from api.models import VegetablePrice

# ==================== 配置读取（替代硬编码） ====================
PROVINCE_CODE = settings.VEGETABLE_CRAWL_CONFIG["PROVINCE_CODE"]
PROVINCE_NAME = settings.VEGETABLE_CRAWL_CONFIG["PROVINCE_NAME"]
TYPE_LIST = settings.VEGETABLE_CRAWL_CONFIG["TYPE_LIST"]
SITE_MIN_DATE = timezone.make_aware(
    datetime.datetime.strptime(settings.VEGETABLE_CRAWL_CONFIG["SITE_MIN_DATE"], "%Y-%m-%d")
).date()
SLEEP_RANGE = settings.VEGETABLE_CRAWL_CONFIG["SLEEP_RANGE"]
TIMEOUT = settings.VEGETABLE_CRAWL_CONFIG["TIMEOUT"]
MAX_PAGE_DEFAULT = settings.VEGETABLE_CRAWL_CONFIG["MAX_PAGE"]
BATCH_SIZE = settings.VEGETABLE_CRAWL_CONFIG["BATCH_SIZE"]
OVERLAP_DEFAULT = settings.VEGETABLE_CRAWL_CONFIG["OVERLAP_DAYS"]
DAYS_DEFAULT = settings.VEGETABLE_CRAWL_CONFIG["DEFAULT_RECENT_DAYS"]
TABLE_INDEX = settings.VEGETABLE_CRAWL_CONFIG["TABLE_INDEX"]

# ==================== 请求头配置 ====================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": f"https://baicai.cnveg.com/price/{PROVINCE_CODE}/",
}


# ==================== 工具函数 ====================
def _to_decimal(x) -> Decimal | None:
    """
    安全转换价格为Decimal类型（保留两位小数）
    处理场景：空值、--、NaN、带￥/逗号的价格
    """
    if x is None:
        return None

    s = str(x).strip()
    # 过滤无效值
    if s in ("", "--", "None", "nan", "NaN"):
        return None

    # 清理价格格式
    s = s.replace("￥", "").replace(",", "")
    try:
        d = Decimal(s)
        return d.quantize(Decimal("0.00"))
    except (InvalidOperation, ValueError):
        return None


def build_url(v_type: str, page: int) -> str:
    """构建爬取URL（对齐旧爬虫规则）"""
    return f"http://{v_type}.cnveg.com/price/{PROVINCE_CODE}/p{page}.html"


def _decode_html(resp: requests.Response) -> str:
    """HTML解码：优先utf-8，兜底gb18030（适配中文网站）"""
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("gb18030", errors="ignore")


def _read_price_table(html_text: str) -> pd.DataFrame:
    """
    读取网页表格：对齐旧逻辑，取指定索引的表格
    :param html_text: 网页源码
    :return: 格式化后的DataFrame，空则返回空DF
    """
    tables = pd.read_html(StringIO(html_text))
    # 改用配置的TABLE_INDEX，不再硬编码16
    if len(tables) <= TABLE_INDEX:
        return pd.DataFrame()

    df = pd.DataFrame(tables[TABLE_INDEX]).copy()
    df.columns = [
        "vegetable_name", "market_name", "min_price",
        "max_price", "avg_price", "date"
    ]
    return df


def fetch_page_df(
        v_type: str,
        page: int,
        session: requests.Session,
        debug: bool = False
) -> pd.DataFrame:
    """
    爬取单页数据并初步清洗
    :param v_type: 蔬菜品类子域名
    :param page: 页码
    :param session: 请求会话
    :param debug: 是否打印调试信息
    :return: 清洗后的DataFrame，空则返回空DF
    """
    url = build_url(v_type, page)
    try:
        resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        if debug:
            print(f"  GET {url} -> 超时（{TIMEOUT}s）")
        return pd.DataFrame()

    # 调试信息输出
    if debug:
        print(
            f"  GET {url} -> {resp.status_code}, "
            f"final_url={resp.url}, len={len(resp.content)}"
        )

    # 非200状态码返回空
    if resp.status_code != 200:
        return pd.DataFrame()

    # 解析表格并清洗
    html_text = _decode_html(resp)
    df = _read_price_table(html_text)

    # 记录URL信息（用于重复页检测）
    df.attrs["requested_url"] = url
    df.attrs["final_url"] = resp.url

    if df.empty:
        return df

    # 数据类型转换 + 空值过滤
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["min_price"] = df["min_price"].apply(_to_decimal)
    df["max_price"] = df["max_price"].apply(_to_decimal)
    df["avg_price"] = df["avg_price"].apply(_to_decimal)
    df = df.dropna(subset=["date"]).copy()

    # 过滤无效蔬菜名称/市场名称（空/特殊字符）
    df = df[
        (df["vegetable_name"].str.strip() != "") &
        (df["market_name"].str.strip() != "")
        ].copy()

    # 过滤异常价格（比如价格≤0 或 价格>100，根据业务调整阈值）
    df = df[
        (df["avg_price"] > 0) & (df["avg_price"] < 100) &
        (df["min_price"] <= df["max_price"])  # 最低价≤最高价（逻辑校验）
        ].copy()

    df["market_name"] = df["market_name"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["province_code"] = PROVINCE_CODE
    df["province_name"] = PROVINCE_NAME
    df["category"] = v_type
    df["crawl_time"] = now()

    return df


# ==================== 核心命令类 ====================
class Command(BaseCommand):
    help = (
        "准实时增量爬取福建蔬菜价格：\n"
        "1. 抓取最近N天数据 + 回补重叠天数\n"
        "2. 自动去重入库，避免重复数据\n"
        "3. 重复页检测，自动停止翻页"
    )

    def add_arguments(self, parser):
        """添加命令行参数（默认值从配置读取）"""
        parser.add_argument(
            "--days",
            type=int,
            default=DAYS_DEFAULT,
            help=f"抓取范围天数（默认{DAYS_DEFAULT}天）"
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=OVERLAP_DEFAULT,
            help=f"回补重叠天数（默认{OVERLAP_DEFAULT}天）"
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=MAX_PAGE_DEFAULT,
            help=f"每个品种最多翻页数（默认{MAX_PAGE_DEFAULT}页）"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="打印调试信息（URL/状态码/响应长度）"
        )

    def handle(self, *args, **options):
        """核心执行逻辑"""
        # 解析命令行参数
        days = max(1, int(options["days"]))
        overlap = max(1, int(options["overlap"]))
        max_pages = max(1, int(options["max_pages"]))
        debug = bool(options.get("debug", False))

        # 计算爬取日期范围
        agg = VegetablePrice.objects.aggregate(max_date=Max("date"))
        db_max_date = agg["max_date"]
        end_date = date.today()

        # 最近days天的起始日期
        recent_start = end_date - timedelta(days=days - 1)
        recent_start = max(recent_start, SITE_MIN_DATE)

        # 叠加回补天数后的起始日期
        start_date = recent_start - timedelta(days=overlap - 1)
        start_date = max(start_date, SITE_MIN_DATE)

        # 基于数据库已有数据调整起始日期（避免漏回补）
        if db_max_date and db_max_date >= SITE_MIN_DATE:
            db_based = db_max_date - timedelta(days=overlap - 1)
            db_based = max(db_based, SITE_MIN_DATE)
            start_date = min(start_date, db_based)

        # 打印爬取范围信息
        self.stdout.write(
            f"\n【爬取配置】\n"
            f"数据库最新日期：{db_max_date}\n"
            f"爬取起始日期：{start_date}\n"
            f"爬取结束日期：{end_date}\n"
            f"参数：days={days}, overlap={overlap}, max_pages={max_pages}"
        )

        # 预加载已有数据的唯一键（避免重复入库）
        existing_keys = set(
            VegetablePrice.objects.filter(
                province_code=PROVINCE_CODE,
                date__gte=start_date,
                date__lte=end_date,
            ).values_list("vegetable_name", "market_name", "date")
        )

        # 初始化请求会话 + 统计变量
        session = requests.Session()
        inserted = 0  # 成功入库数
        skipped = 0  # 重复跳过数
        total_seen = 0  # 过滤后总数据数
        to_create: list[VegetablePrice] = []  # 批量入库列表

        # 遍历每个蔬菜品类
        for v_type in TYPE_LIST:
            self.stdout.write(f"\n========== 开始爬取品类：{v_type} ==========")

            # 重复页检测变量
            last_final_url = None
            last_signature = None
            repeat_streak = 0

            # 遍历页码
            for page in range(1, max_pages + 1):
                # 随机休眠（避免反爬）
                time.sleep(random.uniform(*SLEEP_RANGE))

                # 爬取单页数据
                raw_df = fetch_page_df(v_type, page, session, debug=debug)
                if raw_df.empty:
                    self.stdout.write(f"  页码{page}：数据为空/无效，停止翻页")
                    break

                # 获取URL和日期信息
                requested_url = raw_df.attrs.get("requested_url")
                final_url = raw_df.attrs.get("final_url")
                page_dates = raw_df["date"].tolist()
                page_min_date = min(page_dates)
                page_max_date = max(page_dates)

                # 过滤到目标日期范围
                df = raw_df[
                    (raw_df["date"] >= start_date) &
                    (raw_df["date"] <= end_date)
                    ]

                # 空数据处理
                if df.empty:
                    if page_min_date < start_date:
                        self.stdout.write(f"  页码{page}：数据均早于起始日期，停止翻页")
                        break
                    else:
                        self.stdout.write(f"  页码{page}：过滤后为空（页码最小日期{page_min_date}），继续翻页")
                        continue

                # 页面签名（用于重复页检测）
                uniq_dates = tuple(sorted(set(df["date"].tolist())))
                page_signature = (
                    min(uniq_dates), max(uniq_dates), len(df), uniq_dates
                )

                # 检测重复页（URL或签名重复）
                if (final_url == last_final_url) or (page_signature == last_signature):
                    repeat_streak += 1
                else:
                    repeat_streak = 0

                last_final_url = final_url
                last_signature = page_signature

                if repeat_streak >= 1:
                    self.stdout.write(
                        f"  页码{page}：检测到重复页（请求URL={requested_url}，最终URL={final_url}），停止翻页"
                    )
                    break

                # 遍历数据并去重
                for record in df.to_dict(orient="records"):
                    total_seen += 1
                    record_key = (
                        record["vegetable_name"],
                        record["market_name"],
                        record["date"]
                    )
                    # 重复数据跳过
                    if record_key in existing_keys:
                        skipped += 1
                        continue
                    # 新增数据加入待入库列表
                    existing_keys.add(record_key)
                    to_create.append(VegetablePrice(**record))

                # 批量入库（达到批次阈值）
                if len(to_create) >= BATCH_SIZE:
                    VegetablePrice.objects.bulk_create(to_create, batch_size=BATCH_SIZE)
                    inserted += len(to_create)
                    to_create.clear()

                # 打印页码进度
                self.stdout.write(
                    f"  页码{page}：有效数据{len(df)}条 | 页码日期范围：{page_min_date} ~ {page_max_date}"
                )

                # 数据已早于起始日期，停止翻页
                if page_min_date < start_date:
                    self.stdout.write(f"  页码{page}：已翻到早于起始日期的数据，停止翻页")
                    break

        # 收尾：入库剩余数据
        if to_create:
            VegetablePrice.objects.bulk_create(to_create, batch_size=BATCH_SIZE)
            inserted += len(to_create)
            to_create.clear()

        # 打印最终统计结果
        self.stdout.write(
            self.style.SUCCESS(
                f"\n========== 爬取完成 ==========\n"
                f"成功入库：{inserted} 条\n"
                f"重复跳过：{skipped} 条\n"
                f"过滤后总数据：{total_seen} 条"
            )
        )
