from datetime import datetime

params = {
    'data_dir': './data',
    'price_dir' : './data/price',
    'result_dir' : './result',
    'financial_dir' : './data/financial',
    'factor_dir' : './factors',
    'model_type' : 'dt', # 可选：dt rf xgb

    # 下面的暂时没用
    'start_date': '20180101',
    'end_date':  datetime.today().strftime('%Y%m%d'),
    'stock_list': 'csi500_stock_list.csv',  # 需要放置在data_dir下，股票代码对应的列名为 Code
    'list_code' : '000905', # 中证500
}