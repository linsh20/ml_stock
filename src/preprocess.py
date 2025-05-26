# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd


def format_code_list(x):
    print(x)
    ret = ",".join([f"{str(code).zfill(6)}" for code in x])
    return ret


def daily_data_2_stock_list(): # 从日度数据生成成分股股票列表 暂时不用
    """
    处理 merge_final.parquet：
    1. 生成“有成分股变动”的调仓日列表，输出 zz500_list.csv，
       并额外给出每期相对上一期的新增(add)和剔除(minus)成分股。
    2. 基于该列表做 9 期滑动窗口交集，输出 zz500_list_filter.csv。
    """
    # ======== 常量定义 ========
    file_path = "../data/905_daily_fill.csv"
    PARQUET_PATH = "../data/raw/merge_final.parquet"
    OUT_DIR = "../data/processed"
    FNAME_ALL = "zz500_list.csv"
    FNAME_WIN9 = "zz500_list_filter.csv"

    DATE_COL = "日期"  # parquet 中的日期列
    CODE_COL = "证券代码"  # parquet 中的股票代码列
    # =======================

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(file_path, dtype={'证券代码': str}, encoding='utf-8')
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df.dropna(subset=['日期', '证券代码'], inplace=True)
    df['证券代码'] = df['证券代码'].astype(str)
    # 按日期聚合，得到每个日期的股票代码集合
    daily_stocks_map = df.groupby('日期')['证券代码'].apply(set).to_dict()
    sorted_dates = sorted(daily_stocks_map.keys())

    output_rows = []
    previous_stocks_set = set()

    for current_date in sorted_dates:
        current_stocks_set = daily_stocks_map.get(current_date, set())

        added_stocks = current_stocks_set - previous_stocks_set
        removed_stocks = previous_stocks_set - current_stocks_set

        # 要求1: date只保留有变动的日期
        if added_stocks or removed_stocks:
            count_val = len(current_stocks_set)
            output_rows.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'code': sorted(list(current_stocks_set)),
                'count': count_val,
                'add': sorted(list(added_stocks)),
                'minus': sorted(list(removed_stocks))
            })

        previous_stocks_set = current_stocks_set

    changed_list = pd.DataFrame(output_rows)
    # 筛选掉因为停盘退市的中间零星换成分股
    changed_list = changed_list[changed_list["add"].apply(lambda x: len(x) > 2)].reset_index(drop=True)

    cl = changed_list.copy()  # 过滤后的列表，备份一下，因为后面要调格式

    for col in ["code", "add", "minus"]:
        if col in changed_list.columns:
            changed_list[col] = changed_list[col].apply(format_code_list)
    # 导出 zz500_list.csv
    changed_list.to_csv(
        os.path.join(OUT_DIR, FNAME_ALL),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[1] 已写出 {len(changed_list)} 个有变动的调仓日列表 -> {FNAME_ALL}")

    # -------- 5. 基于“有变动”列表做 9 期滑动窗口并集 --------
    records = []

    for i in range(len(cl) - 8):
        window = cl.iloc[i: i + 9]
        common = set(window.iloc[0]["code"])
        for codes in window["code"].iloc[1:]:
            common |= set(codes) # 取并集
        common = sorted(common)

        train_date = window.iloc[0]["date"]
        test_date = window.iloc[6]["date"]
        buy_date = window.iloc[8]["date"]
        if i + 9 < len(cl):
            end_date = cl.iloc[i + 9]["date"]
        else:
            end_date = window.iloc[8]["date"]  # window.iloc[8] 即第 9 期

        records.append({
            "train_date": train_date,
            "test_date": test_date,
            "buy_date": buy_date,
            "end_date": end_date,
            "count": len(common),
            "code": common,

        })

    win9_df = pd.DataFrame(records)
    if "code" in win9_df.columns:
         win9_df["code"] = win9_df["code"].apply(format_code_list)
    win9_df = win9_df.rename(columns={'code' : 'stock_list'})
    win9_df.to_csv(
        os.path.join(OUT_DIR, FNAME_WIN9),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[2] 已写出 {len(win9_df)} 条滑窗交集结果 -> {FNAME_WIN9}")


# ------- 补全公告日期 -------
# 补全函数
def infer_announcement_date(row):
    if pd.notna(row['公告日期']):
        return row['公告日期']  # 已有公告日期，不补
    report_date = row['财报观察日期']
    if report_date.month == 12:
        return pd.Timestamp(year=report_date.year + 1, month=6, day=30)
    elif report_date.month == 6:
        return pd.Timestamp(year=report_date.year, month=8, day=31)
    elif report_date.month == 3:
        return pd.Timestamp(year=report_date.year, month=4, day=30)
    elif report_date.month == 9:
        return pd.Timestamp(year=report_date.year, month=10, day=31)
    else:
        return pd.NaT  # 如果日期不符合以上任何情况（异常）


def process_daily_season_data(): # 1.merge季度日度和财报公告日数据 2.补全公告日期缺失值
    # !/usr/bin/env python
    # -*- coding: utf-8 -*-
    """
    合并季度表、公告日期与日度行情的脚本
    -------------------------------------------------
    1.  从 ../data/financial 目录批量读取 *_balance.csv，
        追加 '证券代码' 字段后整合为 fin_df
    2.  season_df ← ../data/raw/season_500_0512.parquet
        按 ['证券代码','日期']==['证券代码','报告日'] 左连接 fin_df
        对缺失公告日期的，按最晚日期补全，并将“缺失公告日”列设为1
        得到 '公告日期'，并新增 '公开日期' = 公告日期 + 1 日
    3.  daily_df ← ../data/raw/merge_final.parquet
        按同一证券代码，用 merge_asof(direction='forward')
        将 daily_df['日期'] 与 season_ext['公开日期'] 进行
        最近未来日(≥)匹配，把 season_ext 的所有字段拼接进来
    4.  列重命名：'日期'→'date'，'证券代码'→'code'
        保存到 ../data/processed/merge_data.parquet
    -------------------------------------------------
    运行：python merge_quarterly_daily.py
    """

    # ---------- 0. 统一设置 ----------
    season_path = "../data/raw/season_500_0512.parquet"
    daily_path = "../data/raw/merge_final.parquet"

    # test
    # season_path = "../data/processed/read_season.parquet"
    # daily_path = "../data/processed/read_daily.parquet"

    fin_glob = "../data/financial/*_balance.csv"  # 扫描全部股票
    out_path = "../data/processed/merge_data.parquet"

    # ---------- 1. 读取并整理财报 csv ----------
    fin_frames = []
    for fp in glob.glob(fin_glob):
        # 从文件名提取 6 位代码（假设文件名形如 123456_balance.csv）
        code = os.path.basename(fp).split("_")[0]
        df = pd.read_csv(fp,
                         usecols=['报告日', '公告日期'],
                         parse_dates=['报告日', '公告日期'])
        df["证券代码"] = code
        fin_frames.append(df)

    fin_df = pd.concat(fin_frames, ignore_index=True)

    # ---------- 2. 读取 season 文件并补充公告日期 ----------
    season_df = pd.read_parquet(season_path)
    season_df['日期'] = pd.to_datetime(season_df['日期'])

    # 🆕 新增：重命名 season_df 中的 '日期' 列以避免与 daily_df 中的 '日期' 列冲突
    season_df.rename(columns={'日期': '财报观察日期'}, inplace=True)  # 您可以选择一个更合适的描述性名称

    # ---------- 🔍 merge 前的检查 ----------
    print("🔍 merge 前检查：")
    # 更新检查的列名
    print("season_df['财报观察日期'] 类型：", season_df['财报观察日期'].dtype)
    print("fin_df['报告日'] 类型：", fin_df['报告日'].dtype)

    if season_df['财报观察日期'].isna().any():  # ✅ 修改为新的列名
        print("⚠️ season_df 中 '财报观察日期' 存在缺失值！")  # ✅ 修改为新的列名
    else:
        # 这条打印语句 "✅ season_df 中 '财报观察日期' 无缺失" 已经从您的输出来看是正确的
        print("✅ season_df 中 '财报观察日期' 无缺失")  # ✅ 确认这里也使用了新的列名

    if season_df['财报观察日期'].isna().any():
        print("⚠️ season_df 中 '日期' 存在缺失值！")
    else:
        print("✅ season_df 中 '日期' 无缺失")

    if fin_df['报告日'].isna().any():
        print("⚠️ fin_df 中 '报告日' 存在缺失值！")
    else:
        print("✅ fin_df 中 '报告日' 无缺失")

    # 检查两个字段中的唯一值范围
    print("\n📅 season_df['日期'] 示例（前5个）：", season_df['财报观察日期'].dropna().unique()[:5])
    print("📅 fin_df['报告日'] 示例（前5个）：", fin_df['报告日'].dropna().unique()[:5])

    # 检查能否成功 inner merge（即有效对齐的数据量）
    test_merge = season_df.merge(fin_df, left_on=['证券代码', '财报观察日期'], right_on=['证券代码', '报告日'], how='inner')
    print(f"🔗 inner merge 后匹配成功行数: {len(test_merge)}")
    ##### 检查完毕

    season_ext = (
        season_df
        .merge(fin_df,
               left_on=['证券代码', '财报观察日期'],  # ⚠️ 使用新的列名
               right_on=['证券代码', '报告日'],
               how='left',
               validate='1:1',
               )
        .drop(columns=['报告日'])
    )


    ########## 检查缺失情况##########
    # 在计算完公开日期之后添加：
    total = len(season_ext)
    n_null = season_ext['公告日期'].isna().sum()
    n_notnull = total - n_null

    print(f"📊 season_ext 总行数：{total}")
    print(f"🛑 缺失 '公告日期' 的行数：{n_null}")
    print(f"✅ 不缺失 '公告日期' 的行数：{n_notnull}")

    # 按证券代码分组，展示缺失最多的前几个代码
    null_by_code = (
        season_ext[season_ext['公告日期'].isna()]
        ['证券代码'].value_counts()
    )

    print("\n🛑 各证券代码缺失条数（仅显示前10个）：")
    print(null_by_code.head(10))

    # 示例记录
    print("\n🧪 缺失公告日期的前几行示例：")
    print(season_ext[season_ext['公告日期'].isna()].head())

    # 补全缺失值
    season_ext['公告日期'] = season_ext.apply(infer_announcement_date, axis=1)

    # 加一列 “公开日期” (= 公告日 + 1 天)
    season_ext['公开日期'] = season_ext['公告日期'] + pd.Timedelta(days=1)

    # ---------- 3. 读取 daily 文件并做分组向前匹配 ----------
    daily_df = pd.read_parquet(daily_path)

    if '证券代码' in daily_df.columns:
        daily_df['证券代码'] = daily_df['证券代码'].astype(str).str.zfill(6)
        print(daily_df['证券代码'][0])
    else:
        print("⚠️ 警告: '证券代码' 列在 daily_df 中未找到。")

    daily_df['日期'] = pd.to_datetime(daily_df['日期'])
    season_ext['公开日期'] = pd.to_datetime(season_ext['公开日期'])

    daily_df['日期'] = pd.to_datetime(daily_df['日期'])
    season_ext['公开日期'] = pd.to_datetime(season_ext['公开日期'])

    out_frames = []
    # 要拼接到 daily 上的 season_ext 列（排除 key 列）
    season_cols = [c for c in season_ext.columns
                   if c not in ['证券代码', '公开日期']]

    for code, grp_daily in daily_df.groupby('证券代码', sort=False):
        grp_season = season_ext[season_ext['证券代码'] == code]

        # 组内排序
        grp_daily = grp_daily.sort_values('日期').reset_index(drop=True)
        grp_season = grp_season.sort_values('公开日期').reset_index(drop=True)

        if grp_season.empty:
            # 如果该代码没有任何财报，直接给这些行补 NaN
            # 进的这行
            nan_data = {col: pd.NA for col in season_cols}
            out = grp_daily.assign(**nan_data)
        else:
            if grp_daily.empty:
                print(f"⚠️ daily_df 中 {code} 没有数据")
            elif grp_season.empty:
                print(f"⚠️ season_ext 中 {code} 没有数据")
            else:
                max_daily = grp_daily['日期'].max()
                min_season = grp_season['公开日期'].min()
                print(f"\n🔎 代码 {code}：")
                print(f"📅 daily 日期范围：{grp_daily['日期'].min()} ~ {max_daily}")
                print(f"📅 season 公告公开日范围：{min_season} ~ {grp_season['公开日期'].max()}")

                if max_daily < min_season:
                    print("🛑 所有日度数据都早于公开日期，merge_asof 将全部匹配失败！")
                else:
                    print("✅ 日期范围有交集，merge_asof 应能匹配部分数据")



            out = pd.merge_asof(
                grp_daily,
                grp_season,
                left_on='日期',
                right_on='公开日期',
                direction='forward',
                allow_exact_matches=True,
                suffixes=('','季度数据'),
            )
        out_frames.append(out)

    # 把所有代码拼回一起
    merged = pd.concat(out_frames, ignore_index=True)

    # ---------- 4. 列重命名 & 导出 ----------
    merged.rename(columns={'日期': 'date', '证券代码': 'code'}, inplace = True)
    # 日期格式修正
    for c in ['date',  '公告日期', '公开日期']:
        if c in merged.columns:
            merged[c] = pd.to_datetime(merged[c]).dt.date
    print(merged.columns.tolist())
    merged.to_parquet(out_path, index=False)

    print(f"✅ 处理完毕，文件已保存到: {out_path}")


if __name__ == "__main__":
    daily_data_2_stock_list()
    # process_daily_season_data()