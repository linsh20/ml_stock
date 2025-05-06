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

# 使用示例
input_csv = os.path.join(params['data_dir'], 'merge_data_ret.csv')
output_csv = os.path.join(params['data_dir'], 'read_csv.csv')
field_name = '证券代码'
value = ('2')

filter_csv_by_field(input_csv, output_csv, field_name, value)