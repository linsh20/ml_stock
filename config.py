from datetime import datetime

params = {
    'data_dir': './data',
    'price_dir' : './data/price',
    'financial_dir' : './data/financial',
    'factor_dir' : './factors',
    'start_date': '20180101',
    'end_date':  datetime.today().strftime('%Y%m%d'),
    'stock_list': 'csi500_stock_list.csv',  # 需要放置在data_dir下，股票代码对应的列名为 Code
    'list_code' : '000905', # 中证500
}