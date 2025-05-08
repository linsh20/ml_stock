import pandas as pd
from config import params
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from joblib import Parallel, delayed
from tqdm import tqdm

def label_return(r): # 打标签
    if r >= 0.15:
        return 5
    elif r >= 0.05:
        return 4
    elif r >= 0:
        return 3
    elif r >= -0.15:
        return 2
    elif r < -0.15:
        return 1
    else:
        return 0

def calc_forward_returns(group):
    """
    group 是某只股票的时间序列，包含 '日期' 和 '股票价格'
    添加未来12个月和未来4个月的收益率
    """
    group = group.copy()
    group = group.sort_values('date')

    group['ret_fwd_12m'] = (group['股票价格'].shift(-252) / group['股票价格']) - 1
    group['ret_fwd_4m'] = (group['股票价格'].shift(-84) / group['股票价格']) - 1
    return group


def calc_ret_label(df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['code', 'date'], inplace=True)

    # 按股票分组计算未来收益率
    df = df.groupby('code').apply(calc_forward_returns).reset_index(drop=True)

    # 用未来12个月收益率打标签
    df['label'] = df['ret_fwd_12m'].apply(label_return)

    # 保存结果
    # df.to_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), index=False, encoding='utf-8-sig')
    # print("处理完成，结果已保存为 merge_data_ret.csv")
    print("ret_label计算完成")
    return df


def calc_momentum_factor(df):
    df.sort_values(by=['code', 'date'], inplace=True)
    df['daily_return'] = df.groupby('code')['股票价格'].pct_change()
    # 设置月份间隔
    df['6m_return'] = df.groupby('code')['股票价格'].pct_change(periods=6*21)
    df['11m_return'] = df.groupby('code')['股票价格'].pct_change(periods=11*21)

    df['12m_lagged_return'] = df.groupby('code')['daily_return'].shift(12 * 21)
    df['24m_lagged_return'] = df.groupby('code')['daily_return'].shift(24 * 21)
    return df


def calc_period():
    import pandas as pd
    import os

    # 读取数据
    merge_data = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), parse_dates=['date'])
    merge_data.rename(columns={'Beta3Y_Cov_y':'Beta3Y_Cov', 'Beta3Y_Reg_y':'Beta3Y_Reg'}, inplace=True)
    component_periods = pd.read_csv(os.path.join(params['data_dir'], 'component_periods.csv'),
                                    parse_dates=['begin_date', 'end_date'])
    component_periods.rename(columns={'证券代码': 'code'}, inplace=True)

    # 标记 label、ret6、ret11 的可用性
    merge_data['label_ok'] = merge_data['label'] != 0
    merge_data['ret6_ok'] = merge_data['6m_return'].notnull()
    merge_data['ret11_ok'] = merge_data['11m_return'].notnull()

    # 所有其他要求字段（必须全部非空）
    required_cols = ['6m_return', '11m_return', '总市值',
                     'pe', 'pb', 'ps', '现金流比股价',
                     '净资产收益率A', '每股收益',
                     '资本支出比总市值', '流动比率', 'ocfp', 'capex',
                     'evebit', 'evebitda', '企业价值不含货币资金',
                     '12m_lagged_return', '24m_lagged_return',
                     'Beta3Y_Cov', 'Beta3Y_Reg']

    merge_data['all_fields_ok'] = merge_data[required_cols].notnull().all(axis=1)

    results = []

    for idx, row in component_periods.iterrows():
        code = row['code']
        list_begin = row['begin_date']
        list_end = row['end_date']

        stock_data = merge_data[merge_data['code'] == code]
        if stock_data.empty:
            continue

        # 分别筛选每种字段的有效数据
        label_data = stock_data[stock_data['label_ok']]
        ret6_data = stock_data[stock_data['ret6_ok']]
        ret11_data = stock_data[stock_data['ret11_ok']]
        all_fields_data = stock_data[stock_data['all_fields_ok']]

        # 确保四种数据都存在
        if label_data.empty or ret6_data.empty or ret11_data.empty or all_fields_data.empty:
            continue

        # 取四种数据可用区间的交集
        data_begin = max(label_data['date'].min(),
                         ret6_data['date'].min(),
                         ret11_data['date'].min(),
                         all_fields_data['date'].min())

        data_end = min(label_data['date'].max(),
                       ret6_data['date'].max(),
                       ret11_data['date'].max(),
                       all_fields_data['date'].max())

        # 当前成分期与数据区间交集
        begin_date = max(list_begin, data_begin)
        end_date = min(list_end, data_end)

        if begin_date <= end_date:
            results.append({
                'code': str(code).zfill(6),
                'begin_date': begin_date,
                'end_date': end_date,
                'list_begin_date': list_begin,
                'list_end_date': list_end,
                'data_begin_date': data_begin,
                'data_end_date': data_end,
                'valid_days': (end_date - begin_date).days + 1,
                'list_valid_days': (list_end - list_begin).days + 1,
                'data_valid_days': (data_end - data_begin).days + 1
            })

    final_df = pd.DataFrame(results)
    final_df.to_csv(os.path.join(params['data_dir'], 'filtered_stock_date_range.csv'), index=False)
    print(final_df)


def period2cnt():  # 计算每个交易日起始满足条件的股票数（4年4个月）
    df = pd.read_csv(os.path.join(params['data_dir'], 'filtered_stock_date_range.csv'), parse_dates=['begin_date', 'end_date'])

    # 从 merge_data 中提取真实交易日
    merge_data = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), parse_dates=['date'])
    # merge_data = merge_data.rename(columns={'日期': 'date'})

    trade_days = sorted(merge_data['date'].drop_duplicates())
    trade_days = pd.Series(trade_days)

    window_len = 1092  # 4年4个月 ≈ 1092 个交易日
    max_start_idx = len(trade_days) - window_len

    result = []
    for i in range(max_start_idx):
        start_date = trade_days.iloc[i]
        end_date = trade_days.iloc[i + window_len - 1]

        # 核心修改点：允许多段成分期，统计满足滑窗的“行”数，再取唯一 code 数
        valid_rows = df[(df['begin_date'] <= start_date) & (df['end_date'] >= end_date)]
        unique_codes = valid_rows['code'].unique()
        result.append({'date': start_date, 'stock_count': len(unique_codes)})

    result_df = pd.DataFrame(result)
    result_df.to_csv(os.path.join(params['data_dir'], 'rolling_window_stock_count.csv'), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(result_df['date'], result_df['stock_count'], label='Stocks available for 4Y4M window')
    plt.xlabel('Start Date of Window')
    plt.ylabel('Stock Count')
    plt.title('Stock Count Over Time (Window = 4Y4M)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('stock_count_plot.png')
    plt.show()


def get_date_list():
    rolling_df = pd.read_csv(os.path.join(params['data_dir'], 'rolling_window_stock_count.csv'), parse_dates=['date'])
    stock_df = pd.read_csv(os.path.join(params['data_dir'], 'filtered_stock_date_range.csv'), parse_dates=['begin_date', 'end_date'])

    window_days = 1344  # 4年 + 4个月 = 252*4 + 84*4
    interval_days = 84  # 滚动间隔：每 4 个月
    dates = rolling_df['date'].tolist()
    N = len(dates)

    best_sum = -1
    best_indices = []

    for start in range(interval_days):
        indices = list(range(start, N, interval_days))
        selected = rolling_df.iloc[indices]
        total = selected['stock_count'].sum()
        if total > best_sum:
            best_sum = total
            best_indices = indices

    selected_rows = rolling_df.iloc[best_indices].copy()

    output_list = []
    for _, row in selected_rows.iterrows():
        date = row['date']
        end_date = date + pd.Timedelta(days=window_days)

        # 核心修改点：允许同一股票多段，用多个成分期匹配
        valid_codes = stock_df[(stock_df['begin_date'] <= date) & (stock_df['end_date'] >= end_date)]['code']
        valid_codes = valid_codes.astype(str).apply(lambda x: x.zfill(6)).unique()

        if len(valid_codes) == 0:
            continue

        output_list.append({
            'date': date,
            'stock_count': len(valid_codes),
            'stock_list': ','.join(valid_codes)
        })

    result_df = pd.DataFrame(output_list)
    result_df.to_csv(os.path.join(params['data_dir'], 'best_stock_window_snapshot.csv'), index=False)
    print(result_df[['date', 'stock_count']])


def merge_season_data(merge_data, season_data_path, cols):
    """
    将季频数据（season_data）合并到日频数据（merge_data）：
    对每只股票的每个交易日，匹配该日“向前最近一期”已发布的季报数据。
    """

    # 1. 读入并清洗季频数据
    season_data = pd.read_csv(season_data_path, parse_dates=['日期'])
    season_data.rename(columns={'证券代码': 'code', '日期': 'date'}, inplace=True)
    # 只保留 code、date 和用户指定的 cols
    season_data = season_data[['code', 'date'] + cols].dropna(subset=['date'])
    # 确保日期类型
    season_data['date'] = pd.to_datetime(season_data['date'])

    # 2. 准备日频数据
    df = merge_data.copy()
    # 如果原 df 里列名仍是“日期”“证券代码”，先重命名
    if '日期' in df.columns:
        df.rename(columns={'日期': 'date', '证券代码': 'code'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # 3. 对每只股票单独做 merge_asof
    merged_list = []
    for code, daily in df.groupby('code'):
        # 3.1 本只股票的日频和季频数据
        daily = daily.sort_values('date')
        quarter = season_data[season_data['code'] == code].sort_values('date')

        if quarter.empty:
            # 如果该股无任何季报数据，给所有因子列填 NaN
            for c in cols:
                daily[c] = pd.NA
            merged_list.append(daily)
            continue

        # 3.2 在组内做按日期向后匹配
        tmp = pd.merge_asof(
            daily,
            quarter,
            on='date',
            direction='backward',
        )

        # 3.3 去掉多余的 code 列（如果出现了 code_x/code_y）
        if 'code_y' in tmp.columns:
            tmp.drop(columns=['code_y'], inplace=True)
        if 'code_x' in tmp.columns:
            tmp.rename(columns={'code_x': 'code'}, inplace=True)

        merged_list.append(tmp)

    # 4. 拼回去
    merged = pd.concat(merged_list, ignore_index=True)
    return merged

def create_factors(df):
    """
    构建一个新因子作为两个已有列的比值。
    示例：operating_revenue / total_assets
    """
    df['资本支出比总市值'] = df['资本支出'] / df['总市值']
    df['流动比率'] = df['总资产'] / df['总流动负债']
    df['ocfp'] = df['经营活动现金流量净额'] / df['总资产']
    df['capex'] = df['资本支出'] / df['营业收入']
    df['evebit'] = df['企业价值不含货币资金'] / df['EBIT']
    df['evebitda'] = df['企业价值不含货币资金'] / df['EBITDA']
    return df

def process_stock(code, df, market_df, window_days):
    df_stock = df[df['code'] == code].copy()
    df_stock = df_stock.merge(market_df, on='date', how='left')
    beta_cov_list = []
    beta_reg_list = []

    for current_date in df_stock['date'].dropna().unique():
        window_start = current_date - timedelta(days=window_days)
        df_window = df_stock[(df_stock['date'] >= window_start) & (df_stock['date'] <= current_date)]
        df_window = df_window.dropna(subset=['ret', 'market_ret'])  # ← 改动
        if len(df_window) < 700:
            beta_cov_list.append({'code': code, 'date': current_date, 'Beta3Y_Cov': np.nan})
            beta_reg_list.append({'code': code, 'date': current_date, 'Beta3Y_Reg': np.nan})
            continue

        x = df_window[['market_ret']].values
        y = df_window['ret'].values

        cov = np.cov(y, x[:, 0])[0, 1]
        var = np.var(x[:, 0])
        beta_cov = cov / var if var != 0 else np.nan
        beta_cov_list.append({'code': code, 'date': current_date, 'Beta3Y_Cov': beta_cov})

        model = LinearRegression()
        model.fit(x, y)
        beta_reg = model.coef_[0]
        beta_reg_list.append({'code': code, 'date': current_date, 'Beta3Y_Reg': beta_reg})

    return beta_cov_list, beta_reg_list

def calc_beta_3y_factors(df, n_jobs=-1):
    market_df = pd.read_csv('./data/905_price.csv', parse_dates=['日期'])
    market_df.rename(columns={'日期': 'date', '指数回报率': 'market_ret'}, inplace=True)
    market_df['market_ret'] = market_df['market_ret'].astype(float)

    df.sort_values(by=['code', 'date'], inplace=True)
    df['ret'] = df.groupby('code')['股票价格'].pct_change()

    window_days = 365 * 3
    all_codes = df['code'].unique()

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_stock)(code, df, market_df, window_days)
        for code in tqdm(all_codes, desc="Calculating Beta")
    )

    beta_cov_list = [item for sublist in results for item in sublist[0]]
    beta_reg_list = [item for sublist in results for item in sublist[1]]

    beta_cov_df = pd.DataFrame(beta_cov_list)
    beta_reg_df = pd.DataFrame(beta_reg_list)

    df = df.merge(beta_cov_df, on=['code', 'date'], how='left')
    df = df.merge(beta_reg_df, on=['code', 'date'], how='left')
    return df


# def calc_beta_3y_factors(df):
#     # 加载中证500指数数据
#     market_df = pd.read_csv('./data/905_price.csv', parse_dates=['日期'])
#     market_df.rename(columns={'日期': 'date', '指数回报率': 'market_ret'}, inplace=True)
#     market_df['market_ret'] = market_df['market_ret'].astype(float)
#
#     # 计算个股每日收益率
#     df.sort_values(by=['code', 'date'], inplace=True)
#     df['ret'] = df.groupby('code')['股票价格'].pct_change()
#
#     # 初始化结果
#     beta_cov_list = []
#     beta_reg_list = []
#
#     # 设置三年窗口
#     window_days = 365 * 3
#
#     # 遍历每个股票在每个日期的 beta（3年窗口）
#     for code in df['code'].unique():
#         df_stock = df[df['code'] == code].copy()
#         df_stock = df_stock.merge(market_df, on='date', how='left')
#
#         for current_date in df_stock['date'].dropna().unique():
#             window_start = current_date - timedelta(days=window_days)
#             df_window = df_stock[(df_stock['date'] >= window_start) & (df_stock['date'] <= current_date)].dropna()
#
#             if len(df_window) < 60:
#                 beta_cov_list.append({'code': code, 'date': current_date, 'Beta3Y_Cov': np.nan})
#                 beta_reg_list.append({'code': code, 'date': current_date, 'Beta3Y_Reg': np.nan})
#                 continue
#
#             x = df_window[['market_ret']].values
#             y = df_window['ret'].values
#
#             # 协方差 beta
#             cov = np.cov(y, x[:, 0])[0, 1]
#             var = np.var(x[:, 0])
#             beta_cov = cov / var if var != 0 else np.nan
#             beta_cov_list.append({'code': code, 'date': current_date, 'Beta3Y_Cov': beta_cov})
#
#             # 回归 beta
#             model = LinearRegression()
#             model.fit(x, y)
#             beta_reg = model.coef_[0]
#             beta_reg_list.append({'code': code, 'date': current_date, 'Beta3Y_Reg': beta_reg})
#
#     # 合并结果
#     beta_cov_df = pd.DataFrame(beta_cov_list)
#     beta_reg_df = pd.DataFrame(beta_reg_list)
#     df = df.merge(beta_cov_df, on=['code', 'date'], how='left')
#     df = df.merge(beta_reg_df, on=['code', 'date'], how='left')
#
#     return df


if __name__ == '__main__':
    # df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data.csv'), parse_dates=['日期'])
    # df.rename(columns={'日期': 'date', '证券代码': 'code'}, inplace=True)
    # df = df.dropna(subset=['date'])
    # df = merge_season_data(df, os.path.join(params['data_dir'], 'season_data.csv'),
    #                        cols=['EBIT', 'EBITDA'])
    # print("finish merge season data")
    # df = calc_ret_label(df)  # 必须先算
    # df = calc_momentum_factor(df)
    # print("finish calc momentum factor")
    # df = create_factors(df)
    # print("finish create factors")
    # df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), parse_dates=['date'])
    # df = df[df['code'] == 6].copy()
    # df = calc_beta_3y_factors(df)
    # print("finish calc_beta_3y_factors")
    # df.to_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), index=False, encoding='utf-8-sig')
    # print("已保存df")
    calc_period()
    period2cnt()
    get_date_list()
