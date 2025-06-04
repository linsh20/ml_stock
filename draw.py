import os

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

import src.data_loader as dl
from config import params

# import math # math import is done inside draw_line_fea as per original


MODEL_TYPE = ''
TIME_STAMP = ""
# plt.rcParams['font.family'] = 'Noto Sans CJK SC'

# --- New color definitions for beautification ---
COLOR_BLUE = '#1f77b4'  # Matplotlib default blue
COLOR_ORANGE = '#ff7f0e'  # Matplotlib default orange
COLOR_GREEN = '#2ca02c'  # Matplotlib default green
COLOR_RED = '#d62728'  # Matplotlib default red
COLOR_PURPLE = '#9467bd'  # Matplotlib default purple
COLOR_GREY = '#7f7f7f'  # Matplotlib default grey
COLOR_TEAL = '#17becf'  # Matplotlib default teal
COLOR_LIGHT_BLUE_BOX = '#aec7e8'  # Lighter blue for boxplot fill


# --- End of new color definitions ---

def lag_return():
    # 读取CSV内容
    data = pd.read_csv(os.path.join(params['result_dir'], f"lag_return_time_series_{MODEL_TYPE}_{TIME_STAMP}.csv"),
                       parse_dates=['date'])

    # 设置日期为索引（可选）
    data.set_index('date', inplace=True)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['6m_return'], marker='o', label='6m Return', color=COLOR_BLUE, linestyle='-')
    plt.plot(data.index, data['11m_return'], marker='s', label='11m Return', color=COLOR_ORANGE, linestyle='--')
    plt.title('6-Month vs 11-Month Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(f'./result/fig/lag_return_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def backtest_results():
    ###### 图1： 回测收益对比
    # 读取回测结果数据
    df = pd.read_csv(os.path.join(params['result_dir'], f"backtest_results_{MODEL_TYPE}_{TIME_STAMP}.csv"),
                     parse_dates=['hold_start', 'hold_end'])

    # 获取市场收益数据
    market_df = dl.get_905_price_pd()
    market_df = market_df[['date', 'ret_fwd_6m']].rename(columns={'date': 'hold_start'})
    df = df.merge(market_df, on='hold_start', how='left')

    # 构造周期标签
    df['period_label'] = df['hold_start'].dt.strftime('%y-%m') + ' -> ' + df['hold_end'].dt.strftime('%y-%m')

    # 绘图：共用一个坐标轴
    fig, ax = plt.subplots(figsize=(14, 7))  # 稍微增加图形高度以便有更多空间

    # 柱状图：模型平均收益
    bars = ax.bar(df['period_label'], df['avg_return'],
                  color=[COLOR_GREEN if x >= 0 else COLOR_RED for x in df['avg_return']],
                  label='Model Avg Return',
                  zorder=2)  # 设置zorder确保柱状图在特定层

    # 折线图：市场参考收益
    ax.plot(df['period_label'], df['ret_fwd_6m'], color=COLOR_TEAL, marker='o',
            label='Market 6M Return', zorder=3)  # 设置zorder

    # 添加柱状图标签
    bar_label_fontsize = 7.5  # 稍微调整字号
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval,
                f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top',
                fontsize=bar_label_fontsize, zorder=4)

    # 添加折线图标签
    line_label_fontsize = 7.5  # 稍微调整字号
    line_texts = []  # 用于收集文本对象给 adjustText
    for i, yval in enumerate(df['ret_fwd_6m']):
        if pd.notnull(yval):  # 仅为非NaN值添加文本
            # 文本颜色与折线颜色一致
            txt = ax.text(i, yval, f'{yval:.2%}', ha='center',
                          # va 参数由 adjustText 处理会更好
                          fontsize=line_label_fontsize, color=COLOR_TEAL, zorder=5)
            line_texts.append(txt)

    # 坐标轴设置
    ax.set_ylabel('Return')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['period_label'], rotation=90)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title('Average Return per Test Period with Market 6M Return')
    ax.axhline(0, color='black', lw=0.5, linestyle='--')  # 添加一条 y=0 的参考线

    # 图例
    ax.legend(loc='upper left')

    # 首先应用 tight_layout 调整整体布局
    plt.tight_layout()

    # 然后使用 adjustText 调整折线图标签以避免重叠
    if line_texts:  # 仅当有文本时调用
        adjust_text(line_texts,
                    # ax=ax, # 通常 adjustText 可以自动找到轴
                    # expand_points=(1.2, 1.2), # 稍微增加点周围的空间
                    # expand_text=(1.2, 1.2),   # 稍微增加文本间的空间
                    # force_points=0.2,         # 将文本推离点的力度
                    # force_text=0.3,           # 文本间互相推离的力度
                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.6)  # 给标签添加指引线
                    )

    plt.savefig(f'./result/fig/backtest_results_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()

    ###### 图2：超额收益
    # 新图：超额收益（模型收益 - 市场收益）
    df['excess_return'] = df['avg_return'] - df['ret_fwd_6m']

    fig2, ax2 = plt.subplots(figsize=(14, 6))

    # 柱状图：超额收益
    bars2 = ax2.bar(df['period_label'], df['excess_return'],
                    color=[COLOR_GREEN if x >= 0 else COLOR_RED for x in df['excess_return']],  # Updated colors
                    label='Excess Return')

    # 添加柱状图标签
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval,
                 f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)

    # 坐标轴设置
    ax2.set_ylabel('Excess Return')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['period_label'], rotation=90)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_title('Excess Return (Model - Market) per Test Period')

    # 图例
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./result/fig/backtest_excess_return_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def plot_cumulative_return(risk_free_rate=0.0):
    df = pd.read_csv(os.path.join(params['result_dir'], f"backtest_results_{MODEL_TYPE}_{TIME_STAMP}.csv"),
                     parse_dates=['hold_start', 'hold_end'])

    df = df.sort_values(by='hold_start')
    df['cum_return'] = (1 + df['avg_return']).cumprod()

    # 构造字符串横轴标签
    df['period_label'] = df['hold_start'].dt.strftime('%y-%m') + ' -> ' + df['hold_end'].dt.strftime('%y-%m')

    # 指数累计收益
    market_df = dl.get_905_price_pd()[['date', '收盘指数', 'ret_fwd_6m']].rename(columns={'date': 'hold_start'})
    market_base = market_df.loc[market_df['hold_start'] == df['hold_start'].iloc[0], '收盘指数'].values[0]
    market_df['r'] = market_df['收盘指数'] * (1 + market_df['ret_fwd_6m'])
    market_df = market_df.sort_values('hold_start')

    merged = pd.merge(df[['period_label', 'hold_start', 'cum_return']], market_df, on='hold_start', how='left')
    merged['r'] = merged['r'].ffill()
    merged['index_cum_return'] = merged['r'] / market_base

    # 计算模型绩效
    metrics = compute_performance_metrics(df,
                                          risk_free_rate)  # df here refers to the one from backtest_results.csv for model returns

    # 绘图
    plt.figure(figsize=(14, 6))
    x_labels = merged['period_label']
    model_y = merged['cum_return']
    index_y = merged['index_cum_return']

    plt.plot(x_labels, model_y, label='Model Cumulative Return', marker='o', color=COLOR_BLUE)  # Updated color
    for i, val in enumerate(model_y):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=10)  # Original fontsize

    plt.plot(x_labels, index_y, label='Market Cumulative Return (905)', marker='s', linestyle='--',
             color=COLOR_GREY)  # Updated color
    for i, val in enumerate(index_y):
        plt.text(i, val, f'{val:.2f}', ha='center', va='top', fontsize=10, color='gray')  # Original fontsize and color

    plt.xlabel('Period')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return vs. Market Benchmark')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./result/fig/cumulative_return_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()

    # 展示绩效指标
    textstr = '\n'.join([
        f"Annualized Return: {metrics['Annualized Return']:.2%}",
        f"Volatility:         {metrics['Volatility']:.2%}",
        f"Sharpe Ratio:       {metrics['Sharpe Ratio']:.2f}",
        f"Max Drawdown:       {metrics['Max Drawdown']:.2%}"
    ])
    plt.figure(figsize=(6, 3))
    plt.axis('off')
    plt.text(0.01, 0.5, textstr, fontsize=20, va='center')  # Original fontsize
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
    df.rename(columns={'hold_start': 'date'}, inplace=True)
    df = df.sort_values(by='date')

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['accuracy'], marker='o', linestyle='-', color=COLOR_GREEN)  # Updated color

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
    # df_sorted.boxplot(rot=90, showmeans=True) # Original call
    bp = df_sorted.boxplot(rot=90, showmeans=True, patch_artist=True,  # Added patch_artist for coloring
                           return_type='dict',  # <--- 添加这一行
                           meanprops={'marker': 'D', 'markeredgecolor': 'black',
                                      'markerfacecolor': COLOR_RED},
                           medianprops={'color': COLOR_ORANGE, 'linewidth': 1.5},
                           flierprops={'marker': '.', 'markerfacecolor': COLOR_GREY, 'markeredgecolor': COLOR_GREY,
                                       'alpha': 0.5})

    for box in bp['boxes']:
        box.set_facecolor(COLOR_LIGHT_BLUE_BOX)  # Set box face color
        box.set_edgecolor(COLOR_BLUE)  # Set box edge color
    for whisker in bp['whiskers']:
        whisker.set_color(COLOR_BLUE)
        whisker.set_linestyle('--')
    for cap in bp['caps']:
        cap.set_color(COLOR_BLUE)

    # 添加标题和标签
    plt.title('Box Plot Of Feature Importance (sorted by mean)')
    plt.xlabel('Feature Name')
    plt.ylabel('Feature Importance')
    plt.grid(axis='y', linestyle='--', alpha=0.6)  # Added a subtle grid for y-axis

    plt.tight_layout()
    plt.savefig(f'./result/fig/feature_boxplot_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()

def draw_line_fea():
    import pandas as pd  # Original import location
    import matplotlib.pyplot as plt  # Original import location
    import math  # Original import location

    # 设置中文字体防止乱码
    # plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    df = pd.read_csv(f'./result/feature_importance_time_series_{MODEL_TYPE}_{TIME_STAMP}.csv', parse_dates=['date'])

    # 去除日期列获取特征名
    feature_names = df.columns.drop('date')

    # prepare subplot parameters
    n_features = len(feature_names)
    n_cols = 5  # 每行放5个子图
    n_rows = math.ceil(n_features / n_cols)

    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()  # 展平以方便遍历

    # 逐个特征画图
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        ax.plot(df['date'], df[feature], marker='.', linewidth=1.5,
                color=COLOR_BLUE)  # Updated color, marker, linewidth
        ax.set_title(feature)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle=':', alpha=0.5)  # Added subtle grid to subplots

    # 删除多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 总标题与布局调整
    fig.suptitle('Line Chart: Feature Importance by Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 给总标题留出空间
    plt.savefig(f'./result/fig/feature_lineplot_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def compute_performance_metrics(backtest_df, risk_free_rate='data/BND_TreasYield_filter.csv'):
    import pandas as pd  # Original import location
    import numpy as np  # Original import location

    backtest_df = backtest_df.copy()
    backtest_df['hold_start'] = pd.to_datetime(backtest_df['hold_start'])

    # 区分：是路径还是数值
    if isinstance(risk_free_rate, str):  # 文件路径
        rf_df = pd.read_csv(risk_free_rate)
        rf_df['Trddt'] = pd.to_datetime(rf_df['Trddt'])
        rf_df = rf_df.sort_values('Trddt').set_index('Trddt')
        backtest_df['rf'] = backtest_df['hold_start'].map(
            lambda d: rf_df.loc[rf_df.index <= d, 'Yield'].iloc[-1]
            if not rf_df.loc[rf_df.index <= d].empty else np.nan
        )
    else:  # 单一无风险利率数字（如 0.03）
        backtest_df['rf'] = risk_free_rate

    backtest_df = backtest_df.dropna(subset=['avg_return', 'rf'])

    if len(backtest_df) < 2:
        # Simplified return for less than 2 periods, closer to original implication
        # Original didn't explicitly handle this edge case for all metrics, but this is a minimal safe handling
        _nan_metrics = {
            'Annualized Return': np.nan,
            'Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown': np.nan
        }
        if len(backtest_df) == 1:  # if there's one return, max drawdown is just that return if negative
            ret = backtest_df['avg_return'].iloc[0]
            _nan_metrics['Max Drawdown'] = min(0, ret)
            _nan_metrics['Cumulative Return'] = (1 + ret)  # Original didn't have this key but it's computed below
        else:  # len is 0
            _nan_metrics['Cumulative Return'] = np.nan
        return _nan_metrics

    returns = backtest_df['avg_return'].values
    rfs = backtest_df['rf'].values
    n = len(returns)
    m = 2  # 半年一期
    T = n / m

    cumulative_return = np.prod(1 + returns)
    annualized_return = cumulative_return ** (1 / T) - 1

    volatility = np.std(returns)  # This is periodic volatility
    annualized_volatility = volatility * np.sqrt(m)

    excess_return = returns - rfs
    annualized_excess_return = np.mean(excess_return) * m  # As per original logic
    sharpe_ratio = annualized_excess_return / annualized_volatility if annualized_volatility > 0 else np.nan

    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        'Annualized Return': annualized_return,
        'Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Cumulative Return': cumulative_return
        # Added this key to match previous good version, was implicitly calculated
    }


def compare_top_k():
    ts = TIME_STAMP
    output_file = os.path.join(f"./result/top_k_stocks_{MODEL_TYPE}_{ts}.csv")

    # 读取数据
    df = pd.read_csv(output_file, parse_dates=['hold_start', 'hold_end'])

    # 修复 index 列名
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    df = df.sort_values(by='hold_start')

    # 存储每个 index 的 DataFrame 及其最终累计收益
    index_results = []

    for idx in df['index'].unique():
        sub_df = df[df['index'] == idx].copy()
        sub_df['true_ret'] = pd.to_numeric(sub_df['true_ret'], errors='coerce')
        sub_df = sub_df.sort_values(by='hold_start')
        sub_df['cumulative_return'] = (1 + sub_df['true_ret']).cumprod()

        # 提取该 index 的最终累计收益
        final_cum_ret = sub_df['cumulative_return'].iloc[-1] if not sub_df['cumulative_return'].empty else float('-inf')

        # 记录：index，子数据，最后累计收益
        index_results.append((idx, sub_df, final_cum_ret))

    # 根据最终累计收益降序排序，取前5名
    top_5 = sorted(index_results, key=lambda x: x[2], reverse=True)[:5]

    # 开始绘图
    plt.figure(figsize=(10, 6))

    for idx, sub_df, final_ret in top_5:
        plt.plot(sub_df['hold_start'], sub_df['cumulative_return'], label=f'index={idx}, final={final_ret:.2f}')

    # 图像设置
    plt.title(f'Cumulative Return by Top 5 Index - {MODEL_TYPE}')
    plt.xlabel('Hold Start Date')
    plt.ylabel('Cumulative Return')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'./result/cumulative_return_{MODEL_TYPE}_{ts}.png', dpi=300)
    plt.show()



def draw_all(model_type: str, time_stamp: str):
    global MODEL_TYPE, TIME_STAMP
    MODEL_TYPE = model_type
    TIME_STAMP = time_stamp
    backtest_results()
    plot_cumulative_return()  # Uses default risk_free_rate=0.0 as per original
    label_acc()
    # lag_return()
    draw_box_fea()
    draw_line_fea()
    compare_top_k()


if __name__ == '__main__':
    # 需要修改
    os.makedirs('./result/fig', exist_ok=True)
    MODEL_TYPE = 'dt'
    TIME_STAMP = '20250604_120514'
    draw_all(model_type=MODEL_TYPE, time_stamp=TIME_STAMP)
    # backtest_results()
    # plot_cumulative_return(0)