import pandas as pd
from config import params
import os

def filter_csv_by_field(input_path, output_path, column_name, target_value):
    # 读取CSV文件，尝试适配编码
    df = pd.read_csv(input_path, encoding='utf-8')

    # 清洗字段名
    df.columns = df.columns.str.strip()
    column_name = column_name.strip()

    # 转为字符串进行比对（避免数值科学记数法问题）
    df[column_name] = df[column_name].astype(str)
    target_value = str(target_value)

    # 筛选匹配行
    filtered_df = df[df[column_name] == target_value]

    # 输出结果
    filtered_df.to_csv(output_path, index=False)
    print(f"筛选出 {len(filtered_df)} 行，已保存到 {output_path}")


def filter_parquet_by_field(input_path, output_path, column_name, target_value):
    # 读取CSV文件，尝试适配编码
    df = pd.read_parquet(input_path)

    # 清洗字段名
    df.columns = df.columns.str.strip()
    column_name = column_name.strip()

    # 转为字符串进行比对（避免数值科学记数法问题）
    df[column_name] = df[column_name].astype(str)
    target_value = str(target_value)

    # 筛选匹配行
    filtered_df = df[df[column_name] == target_value]

    # 输出结果
    filtered_df.to_csv(output_path, index=False)
    print(f"筛选出 {len(filtered_df)} 行，已保存到 {output_path}")


def count_and_save_unique_values(input_path, column_name, output_unique_path= None):


    df = pd.read_csv(input_path, encoding='utf-8')
    df.columns = df.columns.str.strip()
    column_name = column_name.strip()

    if column_name not in df.columns:
        print(f"错误：列「{column_name}」不存在于文件中。")
        return

    unique_values = df[column_name].astype(str).str.zfill(6).dropna().unique()
    print(f"字段「{column_name}」共有 {len(unique_values)} 个不同值。")

    # 保存为 CSV 文件
    if (output_unique_path != None):
        unique_df = pd.DataFrame(unique_values, columns=[column_name])
        unique_df.to_csv(output_unique_path, index=False)
        print(f"已将所有唯一值保存到 {output_unique_path}")

def filter_parquet_by_field_2_parquet(input_path, output_path, column_name, target_value):
    # 读取CSV文件，尝试适配编码
    df = pd.read_parquet(input_path)

    # 清洗字段名
    df.columns = df.columns.str.strip()
    column_name = column_name.strip()

    # 转为字符串进行比对（避免数值科学记数法问题）
    df[column_name] = df[column_name].astype(str)
    target_value = str(target_value)

    # 筛选匹配行
    filtered_df = df[df[column_name] == target_value]

    # 输出结果
    filtered_df.to_parquet(output_path, index=False)
    print(f"筛选出 {len(filtered_df)} 行，已保存到 {output_path}")



# 使用示例
input_dir = os.path.join('./data/processed/merge_data_ret.parquet')
output_csv = os.path.join('./data/read_csv/read_ret.csv')
field_name = 'date'
value = ('2024-12-27')

filter_parquet_by_field(input_dir, output_csv, field_name, value)
# count_and_save_unique_values(input_csv, field_name, os.path.join(params['data_dir'], 'read_csv_unique.csv'))
# count_and_save_unique_values(input_csv, field_name, None)