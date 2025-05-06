import akshare as ak

stock_financial_report_sina_df = ak.stock_financial_report_sina(stock="sh600600", symbol="资产负债表")
print(stock_financial_report_sina_df)
stock_financial_report_sina_df.to_csv("stock_financial_report_sina.csv", encoding='utf-8-sig')