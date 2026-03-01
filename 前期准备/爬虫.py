import random
import time
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import os

# 设置列名
COLUMN_NAMES = ['vegetable_name', 'market_name', 'min_price', 'max_price', 'avg_price', 'date', 'province_code', 'timestamp', 'category']

# CSV 文件路径
CSV_FILE = '../Veg/data/fujian_data.csv'

def get_data(v_type, sf, page, session, retry=3):
    url = f'http://{v_type}.cnveg.com/price/{sf}/p{page}.html'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36'
    }

    for attempt in range(retry):
        try:
            resp = session.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"访问失败，状态码：{resp.status_code}")
                time.sleep(3)
                continue

            tables = pd.read_html(StringIO(resp.content.decode('utf-8')))
            if len(tables) <= 16:
                print(f"第 {page} 页表格数量不足，跳过")
                return False

            data = tables[16]
            data = pd.DataFrame(data)

            # 手动设置列名
            data.columns = ['vegetable_name', 'market_name', 'min_price', 'max_price', 'avg_price', 'date']

            # 添加额外列
            data['province_code'] = sf
            data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data['category'] = v_type

            # 写入 CSV
            file_exists = os.path.exists(CSV_FILE)
            data.to_csv(CSV_FILE, index=False, mode='a', header=not file_exists, encoding='utf-8-sig')

            return True

        except Exception as e:
            print(f"尝试 {attempt+1}/{retry} 报错：{e}")
            time.sleep(3)

    return False

# 省份列表
provinces = ['fujian']

# 蔬菜类型列表
type_list = ['baicai', 'qincai', 'bocai', 'woju', 'hg', 'tudou', 'fanqie', 'qiezi']

# 每个品种最大页数
type_pages = {
    'baicai': 65,
    'qincai': 33,
    'bocai': 33,
    'woju': 77,
    'hg': 23,
    'tudou': 22,
    'fanqie': 33,
    'qiezi': 23
}

# 创建 requests Session
session = requests.Session()

# 遍历每个品种
for v_type in type_list:
    total_pages = type_pages.get(v_type, 20)
    print(f"\n开始爬取 {v_type}，共 {total_pages} 页")

    for province in provinces:
        print(f"省份：{province}")

        for page in range(1, total_pages + 1):
            delay = random.randint(5, 10)
            print(f"等待 {delay} 秒...")
            time.sleep(delay)

            success = get_data(v_type, province, page, session)
            if success:
                print(f"{v_type} {province} 第 {page} 页爬取完毕")
            else:
                print(f"{v_type} {province} 第 {page} 页爬取失败")

print("\n所有数据爬取完成！")
print(f"数据已保存到 {CSV_FILE}")

