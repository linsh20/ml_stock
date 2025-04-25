import os
import pandas as pd
import akshare as ak
from config import params
import time

def fetch_concept(): # 获取板块 东财
    concept_board_df = ak.stock_board_concept_name_em()
    print(concept_board_df)
    # 找出包含“中证500”的板块
    csi500_board = concept_board_df[concept_board_df["板块名称"].str.contains("中证500")]
    print(csi500_board)
    # 中证500：BK0701

def con2list(): # 板块->股票列表 （成分股） 东财
    df = ak.stock_board_concept_cons_em(symbol="BK0701")
    print(df)
    df_selected = df[["序号", "代码", "名称"]]
    df_selected.columns = ["Index", "Code", "Name"]
    df_selected.to_csv(params['data_dir'] + "/csi500_stock_list.csv", index=False, encoding="utf-8-sig")
    print("保存成功：csi500_stock_list.csv")

def fetch_daily_price(code: str, re_fetch = False, adjust = ""): # 东财
    if adjust == '':
        file_path = params['data_dir'] + f"/price/normal/{code}.csv"
    elif adjust == 'qfq':
        file_path = params['data_dir'] + f"/price/forw_adj/{code}.csv"
    elif adjust == 'hfq':
        file_path = params['data_dir'] + f"/price/back_adj/{code}.csv"
    else:
        print("adjust参数错误")
        return
    if re_fetch == False and os.path.exists(file_path):
        print(f"{code}.csv 已存在，跳过下载。")
        return
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=params['start_date'],end_date=params['end_date'], adjust=adjust)
    # df = ak.stock_zh_a_daily(symbol=code,  start_date=params['start_date'],end_date=params['end_date'], adjust="")
    # print("列名：", df.columns.tolist())
    df.index = pd.to_datetime(df['日期'])
    df.to_csv(file_path, index=True, encoding="utf-8-sig")
    print(f"finish saving daily price data: {code}")
    return df

def fetch_daily_price_from_list(adjust=""):
    df = pd.read_csv(params['data_dir'] + '/' + params['stock_list'], encoding="utf-8-sig",dtype={"Code": str})
    stock_code_list = df["Code"].tolist()
    i = 0
    for code in stock_code_list:
        i += 1
        print(f"Downloading price {code}... now: {i} / {len(stock_code_list)}")
        fetch_daily_price(code = code, adjust = adjust)

def fetch_financial(code, re_fetch = False, sleep_time = 2.5): # 新浪
    # 利润表
    profit_path = f"{params['data_dir']}/financial/{code}_profit.csv"
    if re_fetch == False and os.path.exists(profit_path):
        print(f"{profit_path} 已存在，跳过下载。")
    else:
        profit = ak.stock_financial_report_sina(stock=code, symbol="利润表")
        profit.to_csv(profit_path, index=False, encoding="utf-8-sig")
        time.sleep(sleep_time)
    # 资产负债表
    balance_path = f"{params['data_dir']}/financial/{code}_balance.csv"
    if re_fetch == False and os.path.exists(balance_path):
        print(f"{balance_path} 已存在，跳过下载。")
    else:
        balance = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
        balance.to_csv(balance_path, index=False, encoding="utf-8-sig")
        time.sleep(sleep_time)
    # 现金流量表
    cash_path = f"{params['data_dir']}/financial/{code}_cashflow.csv"
    if re_fetch == False and os.path.exists(cash_path):
        print(f"{cash_path} 已存在，跳过下载。")
    else:
        cash = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
        cash.to_csv(cash_path, index=False, encoding="utf-8-sig")
        time.sleep(sleep_time)
    print(f"finish saving financial data: {code}")


def fetch_financial_from_list():
    df = pd.read_csv(params['data_dir'] + '/' + params['stock_list'], encoding="utf-8-sig", dtype={"Code": str})
    stock_code_list = df["Code"].tolist()
    i = 0
    while i < len(stock_code_list):
        code = stock_code_list[i]
        print(f"Downloading financial {code}... now: {i + 1} / {len(stock_code_list)}")
        try:
            fetch_financial(code)
            i += 1  # 成功才前进
        except Exception as e:
            print(f"获取财务数据失败: {code}, 错误: {e}")
            print("等待180秒后重试...")
            time.sleep(180)  # 出错后等待，然后再次尝试当前 code


def example():
    fetch_concept()
    fetch_daily_price_from_list()

if __name__ == '__main__':
    # fetch_daily_price_from_list(adjust="hfq")
    # fetch_financial_from_list()
    # df = ak.stock_financial_abstract(symbol="002683")
    # print(df)
    # stock_balance_sheet_by_report_em_df = ak.stock_balance_sheet_by_report_em(symbol="002240")
    # print(stock_balance_sheet_by_report_em_df)
    # fetch_daily_price(code='000905')
    # fetch_daily_price(code='000905', adjust='qfq')
    # fetch_daily_price(code='000905', adjust='hfq')
    fetch_financial(code=601665, re_fetch=True)

