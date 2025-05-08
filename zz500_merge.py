import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from config import params


def change2list(df):  # 从成分股变化数据生成成分股时间数据
    # 按日期排序
    df = df.sort_values(by='日期')

    # 用一个集合维护当前成分股
    current_components = set()

    # 获取所有日期（升序）
    all_dates = df['日期'].sort_values().unique()

    output_rows = []
    output_rows_fill = []
    daily_ops_hold = df['日期'][0]

    # 分组处理每个日期
    for date in all_dates:
        daily_ops = df[df['日期'] == date]
        # if len(output_rows) > 0: # 填充日期
        #     for date1 in pd.date_range(start=date_hold, end=date, freq='D'):
        #         for code in sorted(current_components):
        #             output_rows_fill.append({'日期': date1.strftime('%Y-%m-%d'), '证券代码': code})
        # date_hold = date
        for _, row in daily_ops.iterrows():
            code = row['证券代码'].zfill(6)
            op = row['变动方式']
            if op == 1:
                current_components.add(code)
            elif op == 2:
                current_components.discard(code)
        for code in sorted(current_components):
            output_rows.append({'日期': date.strftime('%Y-%m-%d'), '证券代码': code})
        print(date, "成分股数量 ", len(current_components))
    # 保存到新的 CSV
    result_df = pd.DataFrame(output_rows)
    result_df.to_csv('daily_components.csv', index=False, encoding='utf-8-sig')

    # result_fill_df = pd.DataFrame(output_rows_fill)
    # result_fill_df.to_csv('daily_components_fill.csv', index=False, encoding='utf-8-sig')

    return output_rows


def list2periods(output_rows):
    # 将结果转为 DataFrame 便于处理
    df = pd.DataFrame(output_rows)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(by=['证券代码', '日期']).reset_index(drop=True)

    periods = []
    last_date = {}
    begin_date = {}

    for _, row in df.iterrows():
        code = row['证券代码']
        date = row['日期']

        # 如果首次出现
        if code not in begin_date:
            begin_date[code] = date
        # elif (date - last_date[code]) > timedelta(days=1):  # 出现中断
        #     # 记录上一段区间
        #     periods.append({
        #         '证券代码': code,
        #         'begin_date': begin_date[code].strftime('%Y-%m-%d'),
        #         'end_date': last_date[code].strftime('%Y-%m-%d')
        #     })
        #     begin_date[code] = date  # 重设起始时间

        last_date[code] = date

    # 最后一段别忘了保存
    for code in begin_date:
        periods.append({
            '证券代码': code,
            'begin_date': begin_date[code].strftime('%Y-%m-%d'),
            'end_date': last_date[code].strftime('%Y-%m-%d')
        })

    result_df = pd.DataFrame(periods)
    result_df.to_csv(os.path.join(params['data_dir'], 'component_periods.csv'), index=False, encoding='utf-8-sig')
    return result_df


def change2periods(df):
    df = df.sort_values(by=['证券代码', '日期'])

    # 3. 创建 begin_time 和 end_time 配对
    results = []

    # 当前日期作为默认的 end_time
    today = datetime.today().date()

    # 遍历每个证券代码分组
    for code, group in df.groupby('证券代码'):
        group = group.sort_values(by='日期')
        begin_time = None

        for _, row in group.iterrows():
            if row['变动方式'] == 1:
                begin_time = row['日期'].date()
            elif row['变动方式'] == 2 and begin_time is not None:
                results.append({
                    '证券代码': code,
                    'begin_date': begin_time,
                    'end_date': (row['日期']-pd.Timedelta(days=1)).date()
                })
                begin_time = None

        # 如果还有未匹配的 begin_time
        if begin_time is not None:
            results.append({
                '证券代码': code,
                'begin_date': begin_time,
                'end_date': today
            })

    # 4. 转为 DataFrame 并导出
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(params['data_dir'], "component_periods.csv"), index=False, encoding='utf-8-sig')


def change2periods_last4y4m(df):
    df = df.sort_values(['证券代码', '日期'])
    today = datetime.today().date()
    periods = []

    for code, group in df.groupby('证券代码'):
        group = group.sort_values('日期')
        begin_time = None
        for _, row in group.iterrows():
            ts = row['日期']               # pandas.Timestamp
            if row['变动方式'] == 1:
                begin_time = ts.date()
            elif row['变动方式'] == 2 and begin_time is not None:
                # 先在 Timestamp 上减去一天，再取 .date()
                end_date = (ts - pd.Timedelta(days=1)).date()
                periods.append({
                    '证券代码': code,
                    'begin_date': begin_time,
                    'end_date': end_date
                })
                begin_time = None

        if begin_time is not None:
            periods.append({
                '证券代码': code,
                'begin_date': begin_time,
                'end_date': today
            })

    # 延长 4 年零 4 个月
    for p in periods:
        if p['end_date'] != today:
            p['end_date'] = p['end_date'] + relativedelta(years=4, months=4)

    # 合并重叠区间
    merged = []
    periods_df = pd.DataFrame(periods)
    for code, grp in periods_df.groupby('证券代码'):
        grp = grp.sort_values('begin_date')
        current = None
        for _, row in grp.iterrows():
            if current is None:
                current = row.to_dict()
            else:
                if row['begin_date'] <= current['end_date']:
                    current['end_date'] = max(current['end_date'], row['end_date'])
                else:
                    merged.append(current)
                    current = row.to_dict()
        if current:
            merged.append(current)

    result_df = pd.DataFrame(merged)
    result_df.to_csv(os.path.join(params['data_dir'], "component_periods.csv"),
                     index=False, encoding='utf-8-sig')
    return result_df



if __name__ == '__main__':
    # 读取原始数据
    df = pd.read_csv(params['data_dir'] + '/905.csv', encoding='utf-8', dtype={'证券代码': str})  # 变更日期 指数代码 证券代码 变动方式
    df['日期'] = pd.to_datetime(df['变更日期'])
    df['证券代码'] = df['证券代码'].str.zfill(6)

    # output_rows = change2list(df)
    # list2periods(output_rows)

    change2periods_last4y4m(df)

# def list2intersect(output_rows): # 废弃
#     # 1. 收集所有调整日及对应成分股快照
#     date_to_components = {}
#     for row in output_rows:
#         date = row['日期']
#         code = row['证券代码']
#         date_to_components.setdefault(date, set()).add(code)
#
#     # 2. 对所有调整日按时间排序
#     sorted_dates = sorted(date_to_components.keys())
#     intersection_stats = []
#
#     # 3. 滑动窗口取最近6次的交集
#     for i in range(5, len(sorted_dates)):  # 从第6次开始（下标5）
#         recent_dates = sorted_dates[i - 5:i + 1]  # 取最近6次
#         sets = [date_to_components[d] for d in recent_dates]
#         intersection = set.intersection(*sets)
#         intersection_stats.append({
#             '日期': sorted_dates[i],
#             '交集数量': len(intersection)
#         })
#
#     # 4. 保存结果
#     intersection_df = pd.DataFrame(intersection_stats)
#     intersection_df.to_csv(os.path.join(params['data_dir'],'intersection_last_6_adjustments.csv'), index=False, encoding='utf-8-sig')
