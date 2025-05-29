import pandas as pd
import matplotlib.pyplot as plt
import os
from config import params
import src.data_loader as dl
import main_model
import numpy as np


MODEL_TYPE = ''
TIME_STAMP = ""
# plt.rcParams['font.family'] = 'Noto Sans CJK SC'

def feature_importance():
    # 读取CSV内容
    data = pd.read_csv(os.path.join(params['result_dir'], f"feature_importance_time_series_{MODEL_TYPE}_{TIME_STAMP}.csv"), parse_dates=['date'])

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
    plt.savefig(f'./result/fig/feature_importance_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def backtest_results():
    # 读取回测结果数据
    df = pd.read_csv(os.path.join(params['result_dir'], f"backtest_results_{MODEL_TYPE}_{TIME_STAMP}.csv"),
                     parse_dates=['hold_start', 'hold_end'])

    # 获取市场收益数据
    market_df = dl.get_905_price_pd()
    market_df = market_df[['date', 'ret_fwd_6m']].rename(columns={'date': 'hold_start'})
    df = df.merge(market_df, on='hold_start', how='left')

    # 构造周期标签
    df['period_label'] = df['hold_start'].dt.strftime('%Y-%m-%d') + ' to ' + df['hold_end'].dt.strftime('%Y-%m-%d')

    # 绘图：共用一个坐标轴
    fig, ax = plt.subplots(figsize=(14, 6))

    # 柱状图：模型平均收益
    bars = ax.bar(df['period_label'], df['avg_return'],
                  color=['green' if x >= 0 else 'red' for x in df['avg_return']],
                  label='Model Avg Return')

    # 折线图：市场参考收益
    ax.plot(df['period_label'], df['ret_fwd_6m'], color='blue', marker='o', label='Market 6M Return')

    # 添加柱状图标签
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval,
                f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)

    # 添加折线图标签
    for i, yval in enumerate(df['ret_fwd_6m']):
        ax.text(i, yval, f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8, color='blue')

    # 坐标轴设置
    ax.set_ylabel('Return')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['period_label'], rotation=90)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title('Average Return per Test Period with Market 6M Return')

    # 图例
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./result/fig/backtest_results_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def plot_cumulative_return(risk_free_rate=0.0):
    """
    绘制模型与中证500（905）指数的累计收益曲线，并添加数值标签和绩效指标。
    backtest_df 应包含 ['hold_start', 'avg_return'] 列。
    """
    # 1. 模型累计收益
    df = pd.read_csv(os.path.join(params['result_dir'], f"backtest_results_{MODEL_TYPE}_{TIME_STAMP}.csv"),
                     parse_dates=['hold_start', 'hold_end'])

    df = df.sort_values(by='hold_start')
    df['cum_return'] = (1 + df['avg_return']).cumprod()

    # 2. 指数累计收益
    market_df = dl.get_905_price_pd()[['date', '收盘指数', 'ret_fwd_6m']].rename(columns={'date': 'hold_start'})
    market_base = market_df.loc[market_df['hold_start'] == df['hold_start'].iloc[0], '收盘指数'].values[0]
    market_df['r'] = market_df['收盘指数']*(1+market_df['ret_fwd_6m'])
    market_df = market_df.sort_values('hold_start')
    merged = pd.merge(df[['hold_start', 'cum_return']], market_df, on='hold_start', how='left')
    merged['r'] = merged['r'].ffill()
    merged['index_cum_return'] = merged['r'] / market_base

    # 3. 计算模型绩效指标
    metrics = compute_performance_metrics(df, risk_free_rate)

    # 4. 绘图
    plt.figure(figsize=(12, 6))
    x = merged['hold_start']
    model_y = merged['cum_return']
    index_y = merged['index_cum_return']

    # 模型曲线及标签
    plt.plot(x, model_y, label='Model Cumulative Return', marker='o')
    for i, val in enumerate(model_y):
        plt.text(x[i], val, f'{val:.2f}', ha='center', va='bottom', fontsize=12)

    # 指数曲线及标签
    plt.plot(x, index_y, label='Market Cumulative Return (905)', marker='s', linestyle='--', color='gray')
    for i, val in enumerate(index_y):
        plt.text(x[i], val, f'{val:.2f}', ha='center', va='top', fontsize=12, color='gray')

    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return vs. Market Benchmark')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./result/fig/cumulative_return_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()

    # 5. 显示绩效指标文本框
    textstr = '\n'.join([
        f"Annualized Return: {metrics['Annualized Return']:.2%}",
        f"Volatility:         {metrics['Volatility']:.2%}",
        f"Sharpe Ratio:       {metrics['Sharpe Ratio']:.2f}",
        f"Max Drawdown:       {metrics['Max Drawdown']:.2%}"
    ])
    plt.figure(figsize=(6, 3))
    plt.axis('off')
    plt.text(0.01, 0.5, textstr, fontsize=20, va='center')
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    plt.savefig(f'./result/fig/performance_metrics_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()



def label_acc():
    # 读取CSV文件
    # df = pd.read_csv(f'./result/label_accuracy_{MODEL_TYPE}_{TIME_STAMP}.csv', parse_dates=['date'])
    df = pd.read_csv(os.path.join(params['result_dir'], f"backtest_results_{MODEL_TYPE}_{TIME_STAMP}.csv"),
                     parse_dates=['hold_start', 'hold_end'])
    # 按时间排序（可选）
    df.rename(columns={'hold_start':'date'}, inplace=True)
    df = df.sort_values(by='date')

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['accuracy'], marker='o', linestyle='-')

    # 设置图形标题和坐标轴标签
    plt.title('Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(rotation=45)

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'./result/fig/label_accuracy_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def draw_box_fea():
    # 设置中文字体防止乱码
    # plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    df = pd.read_csv(f'./result/feature_importance_time_series_{MODEL_TYPE}_{TIME_STAMP}.csv')

    # 删除日期列
    df_nodate = df.drop(columns=['date'])

    # 按各列的均值从大到小排序
    mean_sorted_columns = df_nodate.mean().sort_values(ascending=False).index
    df_sorted = df_nodate[mean_sorted_columns]

    # 绘图
    plt.figure(figsize=(14, 8))
    df_sorted.boxplot(rot=90, showmeans=True)

    # 添加标题和标签
    plt.title('Box Plot Of Feature Importance (sorted by mean)')
    plt.xlabel('Feature Name')
    plt.ylabel('Feature Importance')

    plt.tight_layout()
    plt.savefig(f'./result/fig/feature_boxplot_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def draw_line_fea():
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    # 设置中文字体防止乱码
    # plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    df = pd.read_csv(f'./result/feature_importance_time_series_{MODEL_TYPE}_{TIME_STAMP}.csv', parse_dates=['date'])

    # 去除日期列获取特征名
    feature_names = df.columns.drop('date')

    # 准备子图参数
    n_features = len(feature_names)
    n_cols = 5  # 每行放5个子图
    n_rows = math.ceil(n_features / n_cols)

    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()  # 展平以方便遍历

    # 逐个特征画图
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        ax.plot(df['date'], df[feature], marker='o', linewidth=1)
        ax.set_title(feature)
        ax.tick_params(axis='x', rotation=45)

    # 删除多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 总标题与布局调整
    fig.suptitle('Line Chart: Feature Importance by Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 给总标题留出空间
    plt.savefig(f'./result/fig/feature_lineplot_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def compute_performance_metrics(backtest_df, risk_free_rate=0.0):
    backtest_df = backtest_df.copy()

    # 确保日期列是 datetime 格式
    backtest_df['hold_start'] = pd.to_datetime(backtest_df['hold_start'])
    backtest_df['hold_end'] = pd.to_datetime(backtest_df['hold_end'])

    returns = backtest_df['avg_return'].dropna()

    if returns.empty or len(returns) < 2:
        return {
            'Annualized Return': np.nan,
            'Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown': np.nan
        }

    # 使用每行的 hold_end - hold_start 来计算持有期长度，并求平均
    period_days = (backtest_df['hold_end'] - backtest_df['hold_start']).dt.days
    avg_period_days = period_days.mean()
    annual_factor = 365 / avg_period_days if avg_period_days > 0 else 1

    annualized_return = (1 + returns.mean()) ** annual_factor - 1
    annualized_volatility = returns.std() * np.sqrt(annual_factor)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else np.nan

    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        'Annualized Return': annualized_return,
        'Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }


def draw_all(model_type:str, time_stamp:str):
    global MODEL_TYPE, TIME_STAMP
    MODEL_TYPE = model_type
    TIME_STAMP = time_stamp
    backtest_results()
    plot_cumulative_return()
    label_acc()
    feature_importance()
    draw_box_fea()
    draw_line_fea()


if __name__ == '__main__':
    # 需要修改
    os.makedirs('./result/fig', exist_ok=True)
    MODEL_TYPE = 'dt'
    TIME_STAMP = '20250529_232245'
    draw_all(model_type=MODEL_TYPE, time_stamp=TIME_STAMP)
    # backtest_results()
    # plot_cumulative_return(0)
