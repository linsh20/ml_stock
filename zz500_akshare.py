import akshare as ak
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce

def generate_rebalance_dates(start_date: str, end_date: str):
    """生成从 start 到 end 每半年一次的调仓日期（6月30日和12月31日）"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = datetime(start.year, 6 if start.month <= 6 else 12, 30)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=6)
    return dates

def get_csi500_constituents(dates):
    """获取每个调仓日的中证500成分股（000905）"""
    all_components = {}
    for date in dates:
        try:
            # df = ak.index_stock_cons(index=, date=date)
            df = ak.index_detail_hist_cni(symbol="000905", date=date)
            stock_list = df['品种代码'].tolist()
            all_components[date] = set(stock_list)
            print(f"{date} 成分股数量: {len(stock_list)}")
        except Exception as e:
            print(f"获取 {date} 成分股失败：{e}")
    return all_components

def rolling_intersection(component_dict, window_size=5):
    """每 window_size 个调仓日为一个窗口，滚动计算交集"""
    dates = sorted(component_dict.keys())
    result = []

    for i in range(len(dates) - window_size + 1):
        window_dates = dates[i:i + window_size]
        sets_in_window = [component_dict[d] for d in window_dates]
        intersection = reduce(lambda x, y: x & y, sets_in_window)
        result.append({
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
            "stock_count": len(intersection),
            "stocks": sorted(intersection)
        })
    return result

if __name__ == "__main__":
    start_date = "2015-06-30"
    end_date = "2023-12-31"
    window_size = 5  # 每5次调仓滚动

    dates = generate_rebalance_dates(start_date, end_date)
    component_dict = get_csi500_constituents(dates)
    rolling_results = rolling_intersection(component_dict, window_size=window_size)

    print("\n📌 每5次调仓滚动窗口内，始终存在的股票：")
    for idx, res in enumerate(rolling_results):
        print(f"窗口 {idx + 1}: {res['start_date']} ~ {res['end_date']}, 数量: {res['stock_count']}")
        print(res['stocks'])
