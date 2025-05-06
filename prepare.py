import pandas as pd
from config import params
import os
import numpy as np
import matplotlib.pyplot as plt

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
    group = group.sort_values('日期')

    group['ret_fwd_12m'] = (group['股票价格'].shift(-252) / group['股票价格']) - 1
    group['ret_fwd_4m'] = (group['股票价格'].shift(-84) / group['股票价格']) - 1
    return group


def calc_ret_label(df):
    df['日期'] = pd.to_datetime(df['日期'])
    df.sort_values(['证券代码', '日期'], inplace=True)

    # 按股票分组计算未来收益率
    df = df.groupby('证券代码').apply(calc_forward_returns).reset_index(drop=True)

    # 用未来12个月收益率打标签
    df['label'] = df['ret_fwd_12m'].apply(label_return)

    # 保存结果
    # df.to_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), index=False, encoding='utf-8-sig')
    # print("处理完成，结果已保存为 merge_data_ret.csv")
    print("ret_label计算完成")
    return df


def calc_momentum_factor(df):
    df.sort_values(by=['证券代码', '日期'], inplace=True)

    # 设置月份间隔
    df['6m_return'] = df.groupby('证券代码')['股票价格'].pct_change(periods=6*21)
    df['11m_return'] = df.groupby('证券代码')['股票价格'].pct_change(periods=11*21)

    return df


def calc_period(): # 从component_periods.csv 和merge_data_ret.csv 取时间交集
    # 读取数据
    merge_data = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), parse_dates=['日期'])
    component_periods = pd.read_csv(os.path.join(params['data_dir'], 'component_periods.csv'), parse_dates=['begin_date', 'end_date'])

    merge_data = merge_data.rename(columns={
        '日期': 'date',
        '证券代码': 'code',
    })

    # Step 1: 找出每只股票 label != -1 的日期范围，注意，只提取最早和最晚，不考虑中间
    label_range = merge_data[merge_data['label'] != 0].groupby('code')['date'].agg(data_begin_date='min',
                                                                                    data_end_date='max').reset_index()
    ret_6m_range = merge_data[merge_data['6m_return'].notnull()].groupby('code')['date'].agg(data_begin_date='min',
                                                                                     data_end_date='max').reset_index()
    ret_11m_range = merge_data[merge_data['11m_return'].notnull()].groupby('code')['date'].agg(data_begin_date='min',
                                                                                             data_end_date='max').reset_index()

    # Step 2: 获取中证500成分期数据
    component_periods = component_periods.rename(columns={
        '证券代码': 'code',
        'begin_date': 'list_begin_date',
        'end_date': 'list_end_date'
    })

    # Step 3: 合并两个数据集并取日期交集
    merged = pd.merge(component_periods, label_range, on='code', how='inner')
    merged = pd.merge(merged, ret_6m_range, on='code', how='inner')
    merged = pd.merge(merged, ret_11m_range, on='code', how='inner')

    # Step 4: 计算交集
    merged['begin_date'] = merged[['data_begin_date', 'list_begin_date']].max(axis=1)
    merged['end_date'] = merged[['data_end_date', 'list_end_date']].min(axis=1)

    merged['valid_days'] = (merged['end_date'] - merged['begin_date']).dt.days + 1
    merged['list_valid_days'] = (merged['list_end_date'] - merged['list_begin_date']).dt.days + 1
    merged['data_valid_days'] = (merged['data_end_date'] - merged['data_begin_date']).dt.days + 1

    # 只保留有有效交集的记录
    final_result = merged[merged['begin_date'] <= merged['end_date']][[
        'code', 'valid_days', 'list_valid_days', 'data_valid_days',
        'begin_date', 'end_date',
        'data_begin_date', 'data_end_date',
        'list_begin_date', 'list_end_date'
    ]]

    # 输出结果
    final_result.to_csv(os.path.join(params['data_dir'],'filtered_stock_date_range.csv'), index=False)
    print(final_result)


def period2cnt():  # 计算每个交易日起始满足条件的股票数（4年4个月）
    df = pd.read_csv(os.path.join(params['data_dir'],'filtered_stock_date_range.csv'), parse_dates=['begin_date', 'end_date'])

    # 从 merge_data.csv 中提取真实交易日
    merge_data = pd.read_csv(os.path.join(params['data_dir'], 'merge_data.csv'), parse_dates=['日期'])
    merge_data = merge_data.rename(columns={
        '日期': 'date',
    })

    trade_days = sorted(merge_data['date'].drop_duplicates())
    trade_days = pd.Series(trade_days)

    # 设置滑窗长度（4年4个月 ≈ 1092 个交易日）
    window_len = 1092

    # 限制最大起点：防止滑窗溢出
    max_start_idx = len(trade_days) - window_len

    # 统计每个交易日起点对应的股票数
    result = []
    for i in range(max_start_idx):
        start_date = trade_days.iloc[i]
        end_date = trade_days.iloc[i + window_len - 1]

        count = ((df['begin_date'] <= start_date) & (df['end_date'] >= end_date)).sum()
        result.append({'date': start_date, 'stock_count': count})

    # 转成 DataFrame
    result_df = pd.DataFrame(result)

    # 保存到CSV
    result_df.to_csv(os.path.join(params['data_dir'],'rolling_window_stock_count.csv'), index=False)

    # 作图
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


def get_date_list():  # 计算从哪天开始最好，同时输出股票列表
    # 加载数据
    rolling_df = pd.read_csv(os.path.join(params['data_dir'], 'rolling_window_stock_count.csv'), parse_dates=['date'])
    stock_df = pd.read_csv(os.path.join(params['data_dir'], 'filtered_stock_date_range.csv'), parse_dates=['begin_date', 'end_date'])

    # 参数设置
    window_days = 1344 # 252*4+84*4
    interval_days = 84
    dates = rolling_df['date'].tolist()
    N = len(dates)

    # 记录最佳结果
    best_sum = -1
    best_indices = []

    # 遍历所有起点
    for start in range(interval_days):  # 尝试从第 0 到第 83 天作为起点
        indices = list(range(start, N, interval_days))
        selected = rolling_df.iloc[indices]
        total = selected['stock_count'].sum()

        if total > best_sum:
            best_sum = total
            best_indices = indices

    # 最佳选择
    selected_rows = rolling_df.iloc[best_indices].copy()

    # 获取对应股票代码列表
    output_list = []
    for _, row in selected_rows.iterrows():
        date = row['date']
        end_date = date + pd.Timedelta(days=window_days)

        valid_stocks = stock_df[
            (stock_df['begin_date'] <= date) &
            (stock_df['end_date'] >= end_date)
        ]['code'].astype(str).apply(lambda x: x.zfill(6)).tolist()

        if len(valid_stocks) == 0:
            continue  # 跳过股票数为0的日期

        output_list.append({
            'date': date,
            'stock_count': len(valid_stocks),
            'stock_list': ','.join(valid_stocks)
        })

    # 保存输出
    result_df = pd.DataFrame(output_list)
    result_df.to_csv(os.path.join(params['data_dir'], 'best_stock_window_snapshot.csv'), index=False)
    print(result_df[['date', 'stock_count']])



if __name__ == '__main__':
    df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data.csv'), parse_dates=['日期'])
    df = calc_ret_label(df)
    df = calc_momentum_factor(df)
    df.to_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), index=False, encoding='utf-8-sig')
    print("已保存df")
    calc_period()
    period2cnt()
    get_date_list()
