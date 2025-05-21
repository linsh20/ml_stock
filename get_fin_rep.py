import os
import pandas as pd
import akshare as ak
from tqdm import tqdm
import time

# 配置
# csv_path = './data/read_csv_unique.csv'
csv_path = './data/merge_data.csv'
output_dir = './data/financial'

symbol = '资产负债表'  # ← 只下载这一张报表，可改为 '利润表' 或 '现金流量表'
os.makedirs(output_dir, exist_ok=True)

# 添加市场前缀
def convert_stock_code(code):
    code = str(code).zfill(6)
    return code
    if code.startswith(('5', '6', '9')):
        return 'sh' + code
    else:
        return 'sz' + code

# 读取股票代码
stock_df = pd.read_csv(csv_path, usecols = ['证券代码'])
stock_codes = stock_df.iloc[:, 0].dropna().astype(str).str.zfill(6).unique()

# 主循环：检查是否已存在文件，如果没有就下载
for code in tqdm(stock_codes, desc=f"Downloading [{symbol}]"):
    print(f"Downloading [{code}]")
    save_path = os.path.join(output_dir, f'{code}_balance.csv')
    if os.path.exists(save_path):
        continue

    full_code = convert_stock_code(code)

    # 无限重试，直到成功
    while True:
        try:
            df = ak.stock_financial_report_sina(stock=full_code, symbol=symbol)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            time.sleep(3)  # 成功后暂停3秒
            break  # 成功后跳出 while
        except Exception as e:
            print(f"❌ 下载失败: {full_code} - {symbol} - {e}")
            print("⏳ 等待3分钟后重试...")
            time.sleep(180)  # 失败后暂停3分钟再试
