import os
import numpy as np
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
COLOR_ORANGE = '#fd763f'  # Matplotlib default orange
COLOR_YELLOW = '#eeca40'
COLOR_GREEN = '#00a664'  # Matplotlib default green
COLOR_RED = '#d62728'  # Matplotlib default red
COLOR_PURPLE = '#b55489'  # Matplotlib default purple
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


def backtest_results(df, TOP_K=7):

    ts = TIME_STAMP


    # 取第 TOP_K 个组合（注意 index 从 0 开始）
    k = TOP_K - 1
    df_k = df[df['index'] == k].copy()
    df_k = df_k.sort_values(by='hold_start')

    df_k['period_label'] = df_k['hold_start'].dt.strftime('%y-%m') + ' -> ' + df_k['hold_end'].dt.strftime('%y-%m')

    # 图1：回测收益对比
    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(df_k['period_label'], df_k['true_ret_avg'],
                  color=[COLOR_GREEN if x >= 0 else COLOR_RED for x in df_k['true_ret_avg']],
                  label='Model Avg Return',
                  zorder=2)

    ax.plot(df_k['period_label'], df_k['905_true_ret'], color='#1f77b4', marker='o',
            label='Market 6M Return', zorder=3)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval,
                f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top',
                fontsize=7.5, zorder=4)

    line_texts = []
    for i, yval in enumerate(df_k['905_true_ret']):
        if pd.notnull(yval):
            txt = ax.text(i, yval, f'{yval:.2%}', ha='center',
                          fontsize=7.5, color='#1f77b4', zorder=5)
            line_texts.append(txt)

    ax.set_ylabel('Return')
    ax.set_xticks(range(len(df_k)))
    ax.set_xticklabels(df_k['period_label'], rotation=90)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title(f'Average Return per Test Period (TOP_K={TOP_K}) with Market 6M Return')
    ax.axhline(0, color='black', lw=0.5, linestyle='--')
    ax.legend(loc='upper left')

    plt.tight_layout()

    if line_texts:
        adjust_text(line_texts,
                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.6))

    plt.savefig(f'./result/fig/backtest_results_{MODEL_TYPE}_{ts}_top{TOP_K}.png')
    plt.show()

    # 图2：超额收益
    df_k['excess_return'] = df_k['true_ret_avg'] - df_k['905_true_ret']

    fig2, ax2 = plt.subplots(figsize=(14, 6))

    bars2 = ax2.bar(df_k['period_label'], df_k['excess_return'],
                    color=[COLOR_GREEN if x >= 0 else COLOR_RED for x in df_k['excess_return']],
                    label='Excess Return')

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval,
                 f'{yval:.2%}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)

    ax2.set_ylabel('Excess Return')
    ax2.set_xticks(range(len(df_k)))
    ax2.set_xticklabels(df_k['period_label'], rotation=90)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_title(f'Excess Return (Model - Market) per Test Period (TOP_K={TOP_K}, MODEL={MODEL_TYPE})')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(f'./result/fig/backtest_excess_return_{MODEL_TYPE}_{ts}_top{TOP_K}.png')
    plt.show()


def plot_cumulative_return(df, TOP_K=7):
    ts = TIME_STAMP

    # 取指定的 TOP_K 组合数据（注意 index 从 0 开始）
    k = TOP_K - 1
    df_k = df[df['index'] == k].copy().sort_values(by='hold_start')

    df_k['cum_return'] = (1 + df_k['true_ret_avg']).cumprod()
    df_k['market_cum_return'] = (1 + df_k['905_true_ret']).cumprod()

    # 横轴标签
    df_k['period_label'] = df_k['hold_start'].dt.strftime('%y-%m') + ' -> ' + df_k['hold_end'].dt.strftime('%y-%m')

    # 计算绩效指标（使用 true_ret_avg 列）
    metrics = compute_performance_metrics(df_k, TOP_K)

    # 绘图
    plt.figure(figsize=(14, 6))
    x_labels = df_k['period_label']
    model_y = df_k['cum_return']
    market_y = df_k['market_cum_return']

    plt.plot(x_labels, model_y, label='Model Cumulative Return', marker='o', color='orange')
    for i, val in enumerate(model_y):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.plot(x_labels, market_y, label='Market Cumulative Return (905)', marker='s', linestyle='--', color = '#1f77b4')
    for i, val in enumerate(market_y):
        plt.text(i, val, f'{val:.2f}', ha='center', va='top', fontsize=10, color='gray')

    plt.xlabel('Period')
    plt.ylabel('Cumulative Return')
    plt.title(f'Cumulative Return vs. Market Benchmark (TOP_K={TOP_K})')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./result/fig/cumulative_return_{MODEL_TYPE}_{ts}_top{TOP_K}.png')
    plt.show()

    # # 绩效指标图
    # textstr = '\n'.join([
    #     f"Annualized Return: {metrics['Annualized Return']:.2%}",
    #     f"Volatility:         {metrics['Volatility']:.2%}",
    #     f"Sharpe Ratio:       {metrics['Sharpe Ratio']:.2f}",
    #     f"Max Drawdown:       {metrics['Max Drawdown']:.2%}"
    # ])
    # plt.figure(figsize=(6, 3))
    # plt.axis('off')
    # plt.text(0.01, 0.5, textstr, fontsize=20, va='center')
    # plt.title('Model Performance Metrics')
    # plt.tight_layout()
    # plt.savefig(f'./result/fig/performance_metrics_{MODEL_TYPE}_{ts}_top{TOP_K}.png')
    # plt.show()


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
    plt.title(f'Box Plot Of Feature Importance (sorted by mean) ({MODEL_TYPE})')
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
    fig.suptitle(f'Line Chart ({MODEL_TYPE}): Feature Importance by Time', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 给总标题留出空间
    plt.savefig(f'./result/fig/feature_lineplot_{MODEL_TYPE}_{TIME_STAMP}.png')
    plt.show()


def compute_performance_metrics(df_all, TOP_K, risk_free_rate='data/BND_TreasYield_filter.csv'):
    import pandas as pd
    import numpy as np

    k = TOP_K - 1
    df = df_all[df_all['index'] == k].copy()
    df = df.sort_values(by='hold_start')
    df['hold_start'] = pd.to_datetime(df['hold_start'])
    df['avg_return'] = pd.to_numeric(df['true_ret_avg'], errors='coerce')

    if isinstance(risk_free_rate, str):
        rf_df = pd.read_csv(risk_free_rate)
        rf_df['Trddt'] = pd.to_datetime(rf_df['Trddt'])
        rf_df = rf_df.sort_values('Trddt').set_index('Trddt')
        df['rf'] = df['hold_start'].map(
            lambda d: rf_df.loc[rf_df.index <= d, 'Yield'].iloc[-1]
            if not rf_df.loc[rf_df.index <= d].empty else np.nan
        )
    else:
        df['rf'] = risk_free_rate

    df = df.dropna(subset=['avg_return', 'rf'])

    if len(df) < 2:
        _nan_metrics = {
            'Annualized Return': np.nan,
            'Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown': np.nan,
            'Information Ratio': np.nan,
            'Win Rate': np.nan,
            'Calmar Ratio': np.nan
        }
        if len(df) == 1:
            ret = df['avg_return'].iloc[0]
            _nan_metrics['Max Drawdown'] = min(0, ret)
            _nan_metrics['Cumulative Return'] = 1 + ret
        else:
            _nan_metrics['Cumulative Return'] = np.nan
        return _nan_metrics

    returns = df['avg_return'].values
    rfs = df['rf'].values
    n = len(returns)
    m = 2  # 半年一期
    T = n / m
    # 收益
    cumulative_return = np.prod(1 + returns)
    annualized_return = cumulative_return ** (1 / T) - 1
    # 波动率
    volatility = np.std(returns, ddof=1)
    annualized_volatility = volatility * np.sqrt(m)
    # 夏普比
    rfs = rfs / 100
    period_rf = (1 + rfs) ** (1 / m) - 1
    excess_return = returns - period_rf
    mean_excess = np.mean(excess_return)
    sharpe_ratio = (mean_excess / volatility) * np.sqrt(m) if volatility > 0 else np.nan
    # 信息比
    index_df = pd.read_csv('data/905_price.csv')
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.sort_values('date').set_index('date')
    index_df.rename(columns={'收盘指数': 'close'}, inplace=True)
    # 计算基准每期收益，与策略保持一致
    benchmark_returns = []
    for i in range(len(df)):
        d1 = df.iloc[i]['hold_start']
        d2 = df.iloc[i]['hold_end']
        try:
            p1 = index_df.loc[index_df.index <= d1, 'close'].iloc[-1]
            p2 = index_df.loc[index_df.index <= d2, 'close'].iloc[-1]
            ret = (p2 - p1) / p1
        except:
            ret = np.nan
        benchmark_returns.append(ret)

    # df = df.iloc[:-1].copy()
    df['benchmark_return'] = benchmark_returns
    df = df.dropna(subset=['avg_return', 'benchmark_return'])

    # 策略收益 & 基准收益
    returns = df['avg_return'].values
    bench_returns = df['benchmark_return'].values

    # 超额收益
    active_return = returns - bench_returns
    mean_active = np.mean(active_return)
    std_active = np.std(active_return, ddof=1)
    information_ratio = (mean_active / std_active) * np.sqrt(m) if std_active > 0 else np.nan

    returns = np.array(returns)

    # 大于 0 的元素个数
    positive_count = np.sum(returns > 0)

    # 总元素个数
    total_count = len(returns)

    # 正收益占比
    positive_ratio = positive_count / total_count if total_count > 0 else np.nan

    # 打印结果
    print(f"正收益期数：{positive_count}")
    print(f"总期数：{total_count}")
    print(f"胜率（正收益占比）：{positive_ratio:.2%}")
    win_rate = positive_ratio

    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

    result_dict = {
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Information Ratio': information_ratio,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio
    }

    df_result = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Value'])
    df_result.to_csv(f'./result/fig/performance_metrics_{TIME_STAMP}_{MODEL_TYPE}.csv')
    print(f'已保存至 ./result/fig/performance_metrics_{TIME_STAMP}_{MODEL_TYPE}.csv')


def compare_top_k_cum(df, top_index=5):
    ts = TIME_STAMP

    df = df.sort_values(by='hold_start')

    # 存储每个 index 的 DataFrame 及其最终累计收益
    index_results = []

    for idx in df['index'].unique():
        sub_df = df[df['index'] == idx].copy()
        sub_df['true_ret_avg'] = pd.to_numeric(sub_df['true_ret_avg'], errors='coerce')
        sub_df = sub_df.sort_values(by='hold_start')
        sub_df['cumulative_return'] = (1 + sub_df['true_ret_avg']).cumprod()

        # 提取该 index 的最终累计收益
        final_cum_ret = sub_df['cumulative_return'].iloc[-1] if not sub_df['cumulative_return'].empty else float('-inf')

        # 记录：index，子数据，最后累计收益
        index_results.append((idx, sub_df, final_cum_ret))

    # 根据最终累计收益降序排序，取前5名
    top = sorted(index_results, key=lambda x: x[2], reverse=True)[:top_index]

    # 开始绘图
    plt.figure(figsize=(10, 6))

    for idx, sub_df, final_ret in top:
        plt.plot(sub_df['hold_start'].dt.strftime('%y-%m') + ' -> ' + sub_df['hold_end'].dt.strftime('%y-%m'),
                 sub_df['cumulative_return'], label=f'top_k={idx+1}, final={final_ret:.2f}')

    # 图像设置
    plt.title(f'Cumulative Return by Top {top_index} Index - {MODEL_TYPE}')
    plt.xlabel('Hold Period')
    plt.ylabel('Cumulative Return')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'./result/top_k_cumulative_return_{MODEL_TYPE}_{ts}.png', dpi=300)
    plt.show()


def compare_top_k_avg(df, top_index = 5):
    ts = TIME_STAMP

    df = df.sort_values(by='hold_start')

    # 存储每个 index 的 DataFrame 及其正收益期数
    index_results = []

    for idx in df['index'].unique():
        sub_df = df[df['index'] == idx].copy()
        sub_df['true_ret_avg'] = pd.to_numeric(sub_df['true_ret_avg'], errors='coerce')
        sub_df = sub_df.sort_values(by='hold_start')
        sub_df['cumulative_return'] = (1 + sub_df['true_ret_avg']).cumprod()

        # 统计 true_ret_avg > 0 的期数
        positive_count = (sub_df['true_ret_avg'] > 0).sum()

        index_results.append((idx, sub_df, positive_count))

    # 按 true_ret_avg > 0 的期数排序，取前 top_index 个
    top = sorted(index_results, key=lambda x: x[2], reverse=True)[:top_index]

    # 绘图：每期收益率
    plt.figure(figsize=(10, 6))

    for idx, sub_df, pos_count in top:
        x_labels = sub_df['hold_start'].dt.strftime('%y-%m') + ' -> ' + sub_df['hold_end'].dt.strftime('%y-%m')
        plt.plot(x_labels, sub_df['true_ret_avg'], label=f'top_k={idx+1}, >0 count={pos_count}')

    plt.title(f'Period Returns by Top {top_index} Index (Sorted by #Positive Returns) - {MODEL_TYPE}')
    plt.xlabel('Hold Period')
    plt.ylabel('Return per Period')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'./result/top_k_period_return_by_poscount_{MODEL_TYPE}_{ts}.png', dpi=300)
    plt.show()


def compute_index_performance(index_price_path='data/905_price.csv',
                               top_k_df=None,
                               TOP_K=1,
                               rf_path='data/BND_TreasYield_filter.csv'):
    import pandas as pd
    import numpy as np

    # 1. 读取指数数据（中证500）
    price_df = pd.read_csv(index_price_path)
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values('date')
    price_df = price_df.set_index('date')
    price_df.rename(columns={'收盘指数': 'close'}, inplace=True)

    # 2. 提取 TOP_K 对应的持有期起始时间
    k = TOP_K - 1
    df = top_k_df[top_k_df['index'] == k].copy()
    df = df.sort_values('hold_start')
    df['hold_start'] = pd.to_datetime(df['hold_start'])

    # 3. 计算每期收益率（按前后两个起点日期之间的涨跌幅）
    returns = []
    for i in range(len(df)):
        d1 = df.iloc[i]['hold_start']
        d2 = df.iloc[i]['hold_end']  # ✅ 每一期都有自己的持有区间
        try:
            p1 = price_df.loc[price_df.index <= d1, 'close'].iloc[-1]
            p2 = price_df.loc[price_df.index <= d2, 'close'].iloc[-1]
            ret = (p2 - p1) / p1
        except:
            ret = np.nan
        returns.append(ret)

    # df = df.iloc[:-1].copy()
    df['avg_return'] = returns

    # 4. 加载无风险收益率
    rf_df = pd.read_csv(rf_path)
    rf_df['Trddt'] = pd.to_datetime(rf_df['Trddt'])
    rf_df = rf_df.sort_values('Trddt').set_index('Trddt')
    df['rf'] = df['hold_start'].map(
        lambda d: rf_df.loc[rf_df.index <= d, 'Yield'].iloc[-1]
        if not rf_df.loc[rf_df.index <= d].empty else np.nan
    )

    df = df.dropna(subset=['avg_return', 'rf'])

    if len(df) < 2:
        print("样本期数不足，无法计算指标")
        return None

    # 5. 指标计算（和 compute_performance_metrics 中一致）
    returns = df['avg_return'].values
    rfs = df['rf'].values
    n = len(returns)
    m = 2  # 半年一期
    T = n / m

    cumulative_return = np.prod(1 + returns)
    annualized_return = cumulative_return ** (1 / T) - 1

    volatility = np.std(returns, ddof=1)
    annualized_volatility = volatility * np.sqrt(m)

    rfs = rfs / 100
    period_rf = (1 + rfs) ** (1 / m) - 1
    excess_return = returns - period_rf

    mean_excess = np.mean(excess_return)
    std_excess = np.std(excess_return, ddof=1)
    sharpe_ratio = (mean_excess / volatility) * np.sqrt(m) if volatility > 0 else np.nan
    information_ratio = (mean_excess / std_excess) if std_excess > 0 else np.nan

    win_rate = np.mean(returns > 0)

    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    calmar_ratio = (annualized_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-6 else np.nan

    result_dict = {
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Information Ratio': information_ratio,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio
    }

    df_result = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Value'])
    df_result.to_csv(f'./result/fig/performance_metrics_index_{TOP_K}.csv')
    print(f'中证500绩效指标已保存至 ./result/fig/performance_metrics_index_{TOP_K}.csv')

    return df_result



def draw_all(model_type: str, time_stamp: str, top_k: int):
    os.makedirs('./result/fig', exist_ok=True)
    global MODEL_TYPE, TIME_STAMP
    MODEL_TYPE = model_type
    TIME_STAMP = time_stamp
    # 读取组合回测结果
    df = pd.read_csv(os.path.join(f"./result/top_k_stocks_{model_type}_{time_stamp}.csv"),
                     parse_dates=['hold_start', 'hold_end'])
    if 'Unnamed: 0' in df.columns:      # 修复 index 列名（如果有）
        df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)


    backtest_results(df, top_k)
    plot_cumulative_return(df, top_k)  # Uses default risk_free_rate=0.0 as per original
    label_acc()
    # 特征重要性（不改）
    draw_box_fea()
    draw_line_fea()
    # 不同top_k比较
    compare_top_k_cum(df)
    compare_top_k_avg(df)
    compute_index_performance(top_k_df=df)
    print("finish draw all")







if __name__ == '__main__':
    # 需要修改
    os.makedirs('./result/fig', exist_ok=True)


    MODEL_TYPE = 'rf'
    TIME_STAMP = '20250604_185617'
    MODEL_TYPE = 'dt'
    TIME_STAMP = '20250604_165947'
    MODEL_TYPE = 'xgb'
    TIME_STAMP = '20250604_193709'
    draw_all(model_type=MODEL_TYPE, time_stamp=TIME_STAMP, top_k=7)
    # backtest_results()
    # compare_top_k_cum()
    # compare_top_k_avg()
    # backtest_results()
    # plot_cumulative_return(0)