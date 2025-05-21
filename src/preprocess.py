# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd

def daily_data_2_stock_list(): # 从日度数据生成成分股股票列表
    """
    处理 merge_final.parquet：
    1. 生成“有成分股变动”的调仓日列表，输出 zz500_list.csv，
       并额外给出每期相对上一期的新增(add)和剔除(minus)成分股。
    2. 基于该列表做 9 期滑动窗口交集，输出 zz500_list_filter.csv。
    """
    # ======== 常量定义 ========
    PARQUET_PATH = "../data/raw/merge_final.parquet"
    OUT_DIR = "../data/processed"
    FNAME_ALL = "zz500_list.csv"
    FNAME_WIN9 = "zz500_list_filter.csv"

    DATE_COL = "日期"  # parquet 中的日期列
    CODE_COL = "证券代码"  # parquet 中的股票代码列
    # =======================

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- 1. 读 parquet 并去重 --------
    df = pd.read_parquet(PARQUET_PATH, columns=[DATE_COL, CODE_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.drop_duplicates(subset=[DATE_COL, CODE_COL])

    # -------- 2. 生成每期（所有）成分股列表 --------
    all_list = (
        df.groupby(DATE_COL)[CODE_COL]
        .agg(lambda s: sorted(s.unique()))
        .reset_index()
        .rename(columns={DATE_COL: "date", CODE_COL: "code"})
    )
    all_list["count"] = all_list["code"].str.len()

    # -------- 3. 仅保留“有变动”的调仓日 --------
    all_list = all_list.sort_values("date").reset_index(drop=True)
    all_list["changed"] = all_list["code"].ne(all_list["code"].shift())
    changed_list = all_list.loc[all_list["changed"], ["date", "code", "count"]].reset_index(drop=True)

    # -------- 4. 计算新增(add) 和 剔除(minus) 列 --------
    adds, mins = [], []
    for idx, row in changed_list.iterrows():
        curr = set(row["code"])
        if idx == 0:
            prev = set()
        else:
            prev = set(changed_list.at[idx - 1, "code"])
        adds.append(sorted(curr - prev))
        mins.append(sorted(prev - curr))

    changed_list["add"] = adds
    changed_list["minus"] = mins

    # 导出 zz500_list.csv
    changed_list.to_csv(
        os.path.join(OUT_DIR, FNAME_ALL),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[1] 已写出 {len(changed_list)} 个有变动的调仓日列表 -> {FNAME_ALL}")

    # -------- 5. 基于“有变动”列表做 9 期滑动窗口交集 --------
    records = []
    cl = changed_list  # 过滤后的列表
    for i in range(len(cl) - 8):
        window = cl.iloc[i: i + 9]
        common = set(window.iloc[0]["code"])
        for codes in window["code"].iloc[1:]:
            common &= set(codes)
        common = sorted(common)

        begin_date = window.iloc[0]["date"]
        test_date = window.iloc[8]["date"]
        if i + 9 < len(cl):
            end_date = cl.iloc[i + 9]["date"]
        else:
            end_date = window.iloc[8]["date"]  # window.iloc[8] 即第 9 期

        records.append({
            "begin_date": begin_date,
            "test_date": test_date,
            "end_date": end_date,
            "count": len(common),
            "code": common,

        })

    win9_df = pd.DataFrame(records)
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
    report_date = row['日期']
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

    season_ext = (
        season_df
        .merge(fin_df,
               left_on=['证券代码', '日期'],
               right_on=['证券代码', '报告日'],
               how='left',
               validate='1:1',# 一对一匹配，若报错请改成 'm:1'
               )   # ✅ 控制列名后缀！)
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
            nan_data = {col: pd.NA for col in season_cols}
            out = grp_daily.assign(**nan_data)
        else:
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
    merged = merged.rename(columns={'日期': 'date', '证券代码': 'code'})
    # 日期格式修正
    for c in ['date',  '公告日期', '公开日期']:
        if c in merged.columns:
            merged[c] = pd.to_datetime(merged[c]).dt.date
    print(merged.columns.tolist())
    merged.to_parquet(out_path, index=False)

    print(f"✅ 处理完毕，文件已保存到: {out_path}")


if __name__ == "__main__":
    # daily_data_2_stock_list()
    process_daily_season_data()