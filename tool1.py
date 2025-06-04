import pandas as pd
import shutil
import os

# 原始路径
original_file = 'data/processed/merge_data_ret.parquet'
backup_file = 'data/processed/merge_data_ret_raw.parquet'

# Step 1: 备份原始文件
try:
    shutil.copy(original_file, backup_file)
    print(f"✅ 已成功备份文件为：{backup_file}")
except Exception as e:
    print(f"❌ 备份失败：{e}")
    exit()

# Step 2: 读取数据
try:
    df = pd.read_parquet(original_file)
    print(f"✅ 成功读取原始数据，共 {df.shape[0]} 行，{df.shape[1]} 列")
    print("原始数据前5行：")
    print(df.head())
except Exception as e:
    print(f"❌ 读取数据失败：{e}")
    exit()

# Step 3: 删除 code == '002506' 的行
initial_rows = df.shape[0]
df = df[df['code'] != '002506']
deleted_rows = initial_rows - df.shape[0]
print(f"\n✅ 删除 code == '002506' 的行数: {deleted_rows}")
print(f"当前数据总行数: {df.shape[0]}")

# Step 4: 将 label 列值 -1
if 'label' in df.columns:
    before_labels = sorted(df['label'].unique())
    df['label'] = df['label'] - 1
    after_labels = sorted(df['label'].unique())
    print(f"\n✅ label 列已更新")
    print(f"修改前 label 唯一值: {before_labels}")
    print(f"修改后 label 唯一值: {after_labels}")
else:
    print("❌ 未找到 'label' 列，跳过 label 修改步骤。")

# Step 5: 保存处理后的文件
try:
    df.to_parquet(original_file, index=False)
    print(f"\n✅ 已保存处理后的数据到原文件：{original_file}")
except Exception as e:
    print(f"❌ 保存文件失败：{e}")
