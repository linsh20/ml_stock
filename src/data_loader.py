import pandas as pd
import os
from config import params


def get_daily_price_pd(usecols=[]):
    df = pd.read_parquet('./data/merge_data_ret.parquet', columns=usecols)
    # df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), encoding='utf-8-sig', usecols=usecols)
    # if usecols:
    #     df = df[usecols]
    return df


def get_stock_list_pd():
    df = pd.read_csv('data/best_stock_window_snapshot.csv', parse_dates=['date'])
    return df


def csv_2_parquet(file_path=""):
    print(f"\n当前工作目录: {os.getcwd()}")
    print(f"传入的路径: {file_path}")
    print(f"绝对路径: {os.path.abspath(file_path)}")
    print(f"文件是否存在: {os.path.isfile(file_path)}")
    print(f"是否以 .csv 结尾: {file_path.endswith('.csv')}")

    if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
        print(f"无效的 CSV 文件路径: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
        df = df.astype({col: str for col in df.select_dtypes(include=['object']).columns})
        parquet_path = file_path.replace('.csv', '.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"转换成功: {file_path} -> {parquet_path}")
    except Exception as e:
        print(f"转换失败: {file_path}, 错误信息: {e}")


def par_code_2_str(parquet_path="", cols=[""]):
    if not os.path.isfile(parquet_path) or not parquet_path.endswith('.parquet'):
        print(f"❌ 无效的 Parquet 文件路径: {parquet_path}")
        return

    try:
        df = pd.read_parquet(parquet_path)

        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).zfill(6))
            else:
                print(f"⚠️ 列 '{col}' 不存在于文件中，跳过")

        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"✅ 已处理并覆盖保存文件: {parquet_path}")
    except Exception as e:
        print(f"❌ 处理失败: {parquet_path}, 错误信息: {e}")


if __name__ == '__main__':
    os.getcwd()
    # csv_2_parquet('../data/merge_data_ret.csv')
    # csv_2_parquet('../data/raw/season_500_0512.csv')
    # csv_2_parquet(file_path='../data/raw/merge_final.csv')
    # par_code_2_str(parquet_path='../data/raw/merge_final.parquet', cols=['证券代码'])
    par_code_2_str(parquet_path='../data/raw/season_500_0512.parquet', cols=['证券代码'])