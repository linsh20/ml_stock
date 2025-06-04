import pandas as pd
import os
import shutil

def tool():
    # Step 1: 路径定义
    input_path = 'data/processed/merge_data_ret.parquet'
    backup_path = 'data/processed/merge_data_ret_3.parquet'

    # Step 2: 中文转英文列名映射
    col_map = {
        '6m_return': 'momentum_6m',
        '11m_return': 'momentum_11m',
        '净资产收益率A': 'roe',
        'Beta3Y_Reg': 'beta_3Y',
        'Beta3Y_Cov': 'beta_3Y_coef',
        'pb': 'price-to-book',
        'pe': 'earnings-to-price',
        'ps': 'price-to-sales',
        '总市值': 'marketcap',
        '企业价值不含货币资金': 'enterprise-value',
        'evebit': 'evebit',
        'evebitda': 'evebitda',
        '12m_lagged_return': 'returns_12m_lagged_12m',
        '24m_lagged_return': 'returns_12m_lagged_24m',
        '现金流比股价': 'operating cashflow-to-price',
        '资本支出比总市值': 'investment-to-price',
        '每股收益': 'earnings-per-share',
        '流动比率': 'current-ratio',
        'ocfp': 'operating cashflow-to-equity',
        'capex': 'capex'
    }

    # Step 3: 文件备份
    if not os.path.exists(backup_path):
        shutil.copy(input_path, backup_path)
        print(f"备份完成: {backup_path}")
    else:
        print(f"备份已存在: {backup_path}")

    # Step 4: 读取原始数据
    df = pd.read_parquet(input_path)
    print("读取原始数据成功")

    # Step 5: 修改列名
    df.rename(columns=col_map, inplace=True)
    print("列名修改完成")

    # Step 6: 保存结果
    df.to_parquet(input_path, index=False)
    print(f"修改后的文件已保存至: {input_path}")


if __name__ == '__main__':
    tool()