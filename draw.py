import pandas as pd
import matplotlib.pyplot as plt
import os
from config import params

def feature_importance():
    # 读取CSV内容
    data = pd.read_csv(os.path.join(params['result_dir'], 'feature_importance_time_series.csv'), parse_dates=['date'])

    # 设置日期为索引（可选）
    data.set_index('date', inplace=True)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['6m_return'], marker='o', label='6m Return')
    plt.plot(data.index, data['11m_return'], marker='s', label='11m Return')
    plt.title('6-Month vs 11-Month Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def backtest_results():

    # 读取数据
    df = pd.read_csv(os.path.join(params['result_dir'], 'backtest_results.csv'), parse_dates=['test_period_start', 'test_period_end'])

    # 构造周期标签
    df['period_label'] = df['test_period_start'].dt.strftime('%Y-%m-%d') + ' to ' + df['test_period_end'].dt.strftime(
        '%Y-%m-%d')

    # 绘图
    plt.figure(figsize=(14, 6))
    bars = plt.bar(df['period_label'], df['avg_return'], color=['green' if x >= 0 else 'red' for x in df['avg_return']])
    plt.xticks(rotation=90)
    plt.ylabel('Average Return')
    plt.title('Average Return per Test Period')

    # 添加数值标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval,
                 f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top')

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def draw_all():
    feature_importance()
    backtest_results()


if __name__ == '__main__':
    draw_all()