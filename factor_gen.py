from config import params
import pandas as pd
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import numpy as np

RE_CAL_DEFAULT = False # 因子文件存在的情况下是否要重新计算，也可在函数调用时单个修改

import inspect

def prepare_inputs(code: str, factor_name: str):
    """
    根据函数签名动态加载所需数据，并标准化字段命名：
    包括 price_df、market_df、balance_df、profit_df、cashflow_df
    """
    factor_func = factor_functions[factor_name]
    sig = inspect.signature(factor_func)
    required_params = sig.parameters

    inputs = {}

    # 加载价格数据（一定需要）
    price_path = os.path.join(params['price_dir'], factor_adj[factor_name], f'{code}.csv')
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"price file not found: {price_path}")
    if 'price_df' in required_params:
        price_df = pd.read_csv(price_path, encoding='utf-8-sig')
        price_df = standardize_price_df(price_df)
        inputs['price_df'] = price_df

    # 市场指数数据
    if 'market_df' in required_params:
        inputs['market_df'] = load_market_index('000905', factor_adj[factor_name])

    # 资产负债表
    if 'balance_df' in required_params:
        path = os.path.join(params['financial_dir'], f'{code}_balance.csv')
        balance_df = pd.read_csv(path, encoding='utf-8-sig')
        balance_df = standardize_balance_df(balance_df)
        inputs['balance_df'] = balance_df

    # 利润表
    if 'profit_df' in required_params:
        path = os.path.join(params['financial_dir'], f'{code}_profit.csv')
        profit_df = pd.read_csv(path, encoding='utf-8-sig')
        profit_df = standardize_profit_df(profit_df)
        inputs['profit_df'] = profit_df

    # 现金流量表
    if 'cashflow_df' in required_params:
        path = os.path.join(params['financial_dir'], f'{code}_cashflow.csv')
        cashflow_df = pd.read_csv(path, encoding='utf-8-sig')
        cashflow_df = standardize_cashflow_df(cashflow_df)
        inputs['cashflow_df'] = cashflow_df

    # 估值表
    if 'value_df' in required_params:
        path = os.path.join(params['price_dir'], "value", f'{code}.csv')
        value_df = pd.read_csv(path, encoding='utf-8-sig')
        inputs['value_df'] = value_df
    return inputs

def standardize_price_df(df) -> pd.DataFrame:
    rename_map = {
        '当日收盘价': '收盘',
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df

def standardize_balance_df(df):
    rename_map = {
        '总股本': 'total_share',
        '股本': 'total_share',
        '股本总数': 'total_share',
        '实收资本(或股本)': 'total_share',
        '总股数': 'total_share',
        '流通股本': 'total_share',
        '资产总计': 'total_asset',
        '负债合计': 'total_liability',
        '报告日': 'report_date',
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df

def standardize_profit_df(df):
    rename_map = {
        '归属于母公司的净利润': 'net_profit',
        '归属于母公司所有者的净利润' : 'net_profit',
        '营业总收入': 'total_revenue',
        '营业收入': 'revenue',
        '报告日': 'report_date',
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df


def standardize_cashflow_df(df):
    rename_map = {
        '经营活动产生的现金流量净额': 'cashflow_operating',
        '资本支出': 'capital_expenditure',
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df


def calc_from_list(factor_name:str, re_cal = RE_CAL_DEFAULT):
    # step1 获取股票list
    if re_cal == False and os.path.exists(os.path.join(params['factor_dir'], factor_name + '.csv')):
        print(f"{factor_name}.csv 已存在，跳过计算。")
        return
    else:
        print(f"开始生成{factor_name}.csv 。")
    df = pd.read_csv(os.path.join(params['data_dir'], params['stock_list']), encoding="utf-8-sig",dtype={"code": str})
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    stock_code_list = df['Code'].tolist()

    # step2 计算
    all_result = []
    for i, code in enumerate(tqdm(stock_code_list, desc="Calculating")):
        price_path = os.path.join(params['data_dir'], 'price', factor_adj[factor_name], f'{code}.csv')
        if not os.path.exists(price_path):
            print(f'Price file not found: {price_path}')
            continue
        try:
            inputs = prepare_inputs(code, factor_name)  # 统一加载所有输入
            factor_func = factor_functions[factor_name]
            factor_df = factor_func(**inputs).copy()
            factor_df['code'] = code
            factor_df.rename(columns={factor_df.columns[1]: 'factor_value'}, inplace=True)  # 重命名因子列为统一字段
            all_result.append(factor_df)
        except FileNotFoundError as e:
            print(f"[缺文件] {code} skipped: {e}")
        except Exception as e:
            print(f"[出错] {code} skipped: {e}")

    # step3 保存
    if all_result:
        result_df = pd.concat(all_result, ignore_index=True)
        output_path = os.path.join(params['factor_dir'], f'{factor_name}.csv')
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"因子已保存至：{output_path}")
    else:
        print("没有有效因子结果")


def load_market_index(code='000905', adjust = ""):
    """
    加载市场指数数据（如中证500）
    默认使用复权收盘价路径：params['price_dir'] + /back_adj/{code}.csv
    要求 adjust: normal/back_adj/forw_adj
    """
    path = os.path.join(params['price_dir'], adjust , f'{code}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"指数数据文件不存在：{path}")

    df = pd.read_csv(path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['日期'])
    df = df.rename(columns={'收盘': 'close'})
    return df[['date', 'close']]


def calc_momentum_6m(price_df, adjust ='hfq'):
    """
    计算六个月动量因子
    """
    price_df = price_df.rename(columns={'收盘': 'close', '日期': 'date'})
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date')

    price_df['momentum_6m'] = price_df['close'].pct_change(periods=126)
    return price_df[['date', 'momentum_6m']]


def calc_momentum_11m(price_df):
    """
    计算十一个月动量因子
    """
    price_df = price_df.rename(columns={'收盘': 'close', '日期': 'date'})
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date')

    price_df['momentum_11m'] = price_df['close'].pct_change(periods=231)
    return price_df[['date', 'momentum_11m']]


def calc_daily_return(df):
    df = df.sort_values('date')
    df['return'] = df['close'].pct_change()
    return df[['date', 'return']]


def calc_beta_reg(price_df, market_df):
    """
    使用线性回归计算 beta（滚动756日窗口）
    个股收益率 = α + β × 市场收益率 + ε
    如果不足756日则返回NaN
    """
    price_df = price_df.rename(columns={'收盘': 'close', '日期': 'date'})
    price_df['date'] = pd.to_datetime(price_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])

    stock_ret = calc_daily_return(price_df)
    market_ret = calc_daily_return(market_df)
    merged = pd.merge(stock_ret, market_ret, on='date', how='inner', suffixes=('_stock', '_market'))

    if merged.shape[0] < 756:
        return pd.DataFrame({
            'date': merged['date'],
            'beta_reg': np.nan
        })

    beta_list = []
    dates = merged['date'].tolist()

    for i in range(756, len(merged)):
        window = merged.iloc[i - 756:i]
        x = window[['return_market']].values
        y = window['return_stock'].values
        if np.isnan(x).any() or np.isnan(y).any():
            beta = np.nan
        else:
            model = LinearRegression().fit(x, y)
            beta = model.coef_[0]
        beta_list.append({'date': dates[i], 'beta_reg': beta})

    return pd.DataFrame(beta_list)


def calc_beta_cov(price_df, market_df):
    """
        使用协方差公式计算 beta（滚动756日窗口）
        β = Cov(个股收益率, 市场收益率) / Var(市场收益率)
        如果不足756日则返回NaN
    """
    price_df = price_df.rename(columns={'收盘': 'close', '日期': 'date'})
    price_df['date'] = pd.to_datetime(price_df['date'])
    market_df['date'] = pd.to_datetime(market_df['date'])

    stock_ret = calc_daily_return(price_df)
    market_ret = calc_daily_return(market_df)
    merged = pd.merge(stock_ret, market_ret, on='date', how='inner', suffixes=('_stock', '_market'))

    if merged.shape[0] < 756:
        # 返回全为 NaN 的结构，方便合并
        return pd.DataFrame({
            'date': merged['date'],
            'beta_cov': np.nan
        })

    beta_list = []
    dates = merged['date'].tolist()

    for i in range(756, len(merged)):
        window = merged.iloc[i - 756:i]
        cov = window['return_stock'].cov(window['return_market'])
        var = window['return_market'].var()
        beta = cov / var if var != 0 else np.nan
        beta_list.append({'date': dates[i], 'beta_cov': beta})

    return pd.DataFrame(beta_list)


def calc_total_mv(value_df):
    """
    总市值 = 股价 × 总股本
    """
    value_df = value_df.rename(columns={'数据日期': 'date'})
    return value_df[['date', '总市值']]


def calc_pe_inverse(value_df, profit_df, price_df):
    """
    市盈率倒数 = 每股收益 / 股价，EPS = 净利润 / 总股本
    """
    total_mv_df = calc_total_mv(value_df)
    profit_df['report_date'] = pd.to_datetime(profit_df['report_date'])
    total_mv_df['date'] = pd.to_datetime(total_mv_df['date'])
    merged = pd.merge(total_mv_df, profit_df, left_on='date', right_on='report_date', how='outer')
    merged['eps'] = merged['net_profit'] / merged['总市值']
    price_df['日期'] = pd.to_datetime(price_df['日期'])
    merged_price = price_df.merge(merged[['date', 'eps']], left_on='日期', right_on='date', how='left')
    merged_price['eps'] = merged_price['eps'].ffill()
    merged_price['pe_inverse'] = merged_price['eps'] / merged_price['收盘']

    return merged_price[['date', 'pe_inverse']]


def calc_pb(value_df, balance_df, price_df):
    """
    市净率 = 总市值 / 净资产 = 总市值 / (总资产 - 总负债)
    """
    total_mv_df = calc_total_mv(value_df)
    balance_df['report_date'] = pd.to_datetime(balance_df['report_date'])
    total_mv_df['date'] = pd.to_datetime(total_mv_df['date'])

    balance_df['net_asset'] = balance_df['total_asset'] - balance_df['total_liability']
    merged = pd.merge(total_mv_df, balance_df[['report_date', 'net_asset']], left_on='date', right_on='report_date', how='outer')

    price_df['日期'] = pd.to_datetime(price_df['日期'])
    merged_price = price_df.merge(merged[['date', 'net_asset', '总市值']], left_on='日期', right_on='date', how='left')

    merged_price['net_asset'] = merged_price['net_asset'].ffill()
    merged_price['总市值'] = merged_price['总市值'].ffill()
    merged_price['pb'] = merged_price['总市值'] / merged_price['net_asset']

    return merged_price[['date', 'pb']]



def calc_ps(value_df, profit_df, price_df):
    """
    市销率 = 总市值 / 营业收入
    """
    total_mv_df = calc_total_mv(value_df)
    profit_df['report_date'] = pd.to_datetime(profit_df['report_date'])
    total_mv_df['date'] = pd.to_datetime(total_mv_df['date'])

    merged = pd.merge(total_mv_df, profit_df[['report_date', 'revenue']], left_on='date', right_on='report_date', how='outer')

    price_df['日期'] = pd.to_datetime(price_df['日期'])
    merged_price = price_df.merge(merged[['date', 'revenue', '总市值']], left_on='日期', right_on='date', how='left')

    merged_price['revenue'] = merged_price['revenue'].ffill()
    merged_price['总市值'] = merged_price['总市值'].ffill()
    merged_price['ps'] = merged_price['总市值'] / merged_price['revenue']

    return merged_price[['date', 'ps']]



# 所有因子函数接口
factor_functions = {
    'momentum_6m': calc_momentum_6m, # 六个月动量因子
    'momentum_11m': calc_momentum_11m, # 十一个月动量因子
    'beta_cov': calc_beta_cov, # 三年数据计算的协方差Cov
    'beta_reg': calc_beta_reg, # 三年数据计算的线性回归Beta
    'total_mv': calc_total_mv, # 总市值
    'pb' : calc_pb, # 市净率
    'ps' : calc_ps, # 市销率
    'pe_inv': calc_pe_inverse, # 市盈率的倒数
}

# 价格数据复权情况
factor_adj = {
    'momentum_6m': 'back_adj', # 六个月动量因子
    'momentum_11m': 'back_adj', # 十一个月动量因子
    'beta_cov': 'back_adj', # 三年数据计算的协方差Cov
    'beta_reg': 'back_adj', # 三年数据计算的线性回归Beta
    'total_mv': 'forw_adj',  # 总市值
    'pb': 'back_adj',  # 市净率
    'ps': 'back_adj',  # 市销率
    'pe_inv': 'forw_adj' # 市盈率的倒数
}

if __name__ == '__main__':
    os.makedirs(params['factor_dir'], exist_ok=True)
    calc_from_list('momentum_6m')
    calc_from_list('momentum_11m')
    calc_from_list('beta_cov')
    calc_from_list('beta_reg')
    calc_from_list('total_mv')
    calc_from_list('pb', re_cal=True)
    calc_from_list('ps', re_cal=True)
    # calc_from_list('pe_inv', re_cal=True)

