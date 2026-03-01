import pandas as pd
import os

PROVINCE_MAPPING = {'fujian': '福建'}

# 输入输出路径
INPUT_FILE = '../pythonProject1/data/fujiandata.csv'
OUTPUT_FILE = '../pythonProject1/data/fujian_clean.csv'


def clean_data():
    if not os.path.exists(INPUT_FILE):
        print("错误：找不到原始数据文件")
        return

    # 1. 加载数据
    # 指定列名，确保与爬虫存入的一致
    cols = ['vegetable_name', 'market_name', 'min_price', 'max_price', 'avg_price', 'date', 'province_code',
            'timestamp', 'category']
    df = pd.read_csv(INPUT_FILE, names=cols, header=1)

    # 2. 清洗价格列
    # 使用 pd.to_numeric(errors='coerce') 将无法转换的脏数据（如“--”或空值）变为 NaN
    price_cols = ['min_price', 'max_price', 'avg_price']

    for col in price_cols:
        df[col] = df[col].replace({'￥': '', ',': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删掉价格为空的行（因为这种数据没有分析价值）
    df.dropna(subset=['avg_price'], inplace=True)

    # 3. 处理日期和时间
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # 4. 省份转换
    # 用 map 映射，如果不匹配则保留原值或设为未知
    df['province_name'] = df['province_code'].map(PROVINCE_MAPPING)
    df['province_name'] = df['province_name'].fillna('未知')

    # 5. 去重与缺失值处理
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicate_count = initial_count - len(df)
    print(f"去除完全重复的行: {duplicate_count} 条")

    # 填充其余字符串类型的缺失值
    df.fillna('未知', inplace=True)

    # 6. 保存结果
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"清洗完成！处理后数据共 {len(df)} 行")


if __name__ == '__main__':
    clean_data()