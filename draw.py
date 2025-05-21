import pandas as pd
import matplotlib.pyplot as plt
import os
from config import params

def feature_importance():
    # 读取CSV内容
    data = pd.read_csv(os.path.join(params['result_dir'], f"feature_importance_time_series_{params['model_type']}.csv"), parse_dates=['date'])

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
    df = pd.read_csv(os.path.join(params['result_dir'], f"backtest_results_{params['model_type']}.csv"), parse_dates=['test_period_start', 'test_period_end'])

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

def label_acc():
    # 读取CSV文件
    df = pd.read_csv('./result/label_accuracy_dt.csv', parse_dates=['date'])

    # 按时间排序（可选）
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
    plt.show()


def draw_box_fea():
    # 设置中文字体防止乱码
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    df = pd.read_csv('./result/feature_importance_time_series_dt.csv')

    # 删除日期列
    df_nodate = df.drop(columns=['date'])

    # 按各列的均值从大到小排序
    mean_sorted_columns = df_nodate.mean().sort_values(ascending=False).index
    df_sorted = df_nodate[mean_sorted_columns]

    # 绘图
    plt.figure(figsize=(14, 8))
    df_sorted.boxplot(rot=90, showmeans=True)

    # 添加标题和标签
    plt.title('各特征的重要性分布箱型图（按平均值排序）')
    plt.xlabel('特征名称')
    plt.ylabel('特征重要性')

    plt.tight_layout()
    plt.show()

def draw_all():
    feature_importance()
    backtest_results()
def draw_line_fea():
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    # 设置中文字体防止乱码
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    df = pd.read_csv('./result/feature_importance_time_series_dt.csv', parse_dates=['date'])

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
    fig.suptitle('各特征的重要性时间折线图', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 给总标题留出空间
    plt.show()


if __name__ == '__main__':
    # draw_all()
    # label_acc()
    # draw_box_fea()
    draw_line_fea()