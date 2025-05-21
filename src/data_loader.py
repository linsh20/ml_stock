import pandas as pd
import os
from config import params


def get_daily_price_pd(usecols=[]):
    df = pd.read_parquet('./data/merge_data_ret.parquet')
    # df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), encoding='utf-8-sig', usecols=usecols)
    if usecols:
        df = df[usecols]
    return df


def get_stock_list_pd():
    df = pd.read_csv('data/best_stock_window_snapshot.csv', parse_dates=['date'])
    return df


def csv_2_parquet(file_path=""):
    if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
        print(f"无效的 CSV 文件路径: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        parquet_path = file_path.replace('.csv', '.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"转换成功: {file_path} -> {parquet_path}")
    except Exception as e:
        print(f"转换失败: {file_path}, 错误信息: {e}")


if __name__ == '__main__':
    os.getcwd()
    csv_2_parquet('../data/merge_data_ret.csv')