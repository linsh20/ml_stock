# -*- coding: utf-8 -*-
"""
处理 merge_final.parquet：
1. 生成“有成分股变动”的调仓日列表  -> ../data/processed/zz500_list.csv
2. 基于该列表做 9 期滑动窗口交集 -> ../data/processed/zz500_list_filter.csv
"""
import os
import pandas as pd

# ======== 常量定义 ========
PARQUET_PATH = "../data/raw/merge_final.parquet"
OUT_DIR      = "../data/processed"
FNAME_ALL    = "zz500_list.csv"
FNAME_WIN9   = "zz500_list_filter.csv"

DATE_COL = "日期"       # parquet 中的日期列
CODE_COL = "证券代码"   # parquet 中的股票代码列
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

# -------- 3. 仅保留“成分股有变动”的调仓日 --------
#   中证500半年调一次仓，但若某次调仓里成分股未变，则不输出
all_list = all_list.sort_values("date").reset_index(drop=True)
all_list["changed"] = all_list["code"].ne(all_list["code"].shift())
changed_list = (
    all_list.loc[all_list["changed"], ["date", "code", "count"]]
            .reset_index(drop=True)
)
# 导出结果
changed_list.to_csv(
    os.path.join(OUT_DIR, FNAME_ALL),
    index=False,
    encoding="utf-8-sig"
)
print(f"[1] 已写出 {len(changed_list)} 个有变动的调仓日列表 -> {FNAME_ALL}")

# -------- 4. 基于“有变动”列表做 9 期滑动窗口交集 --------
records = []
cl = changed_list  # 过滤后的列表
for i in range(len(cl) - 8):
    window = cl.iloc[i : i + 9]
    common = set(window.iloc[0]["code"])
    for codes in window["code"].iloc[1:]:
        common &= set(codes)
    common = sorted(common)

    begin_date = window.iloc[0]["date"]
    test_date  = window.iloc[8]["date"]
    end_date   = cl.iloc[i + 9]["date"] if i + 9 < len(cl) else pd.NaT

    records.append({
        "begin_date": begin_date,
        "test_date" : test_date,
        "end_date"  : end_date,
        "code"      : common,
        "count"     : len(common),
    })

win9_df = pd.DataFrame(records)
win9_df.to_csv(
    os.path.join(OUT_DIR, FNAME_WIN9),
    index=False,
    encoding="utf-8-sig"
)
print(f"[2] 已写出 {len(win9_df)} 条滑窗交集结果 -> {FNAME_WIN9}")
