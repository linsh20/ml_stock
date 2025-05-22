import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from config import params
from sklearn.impute import SimpleImputer
import time
from datetime import datetime
from tqdm import tqdm
import draw
import csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from src import data_loader


def format_seconds(seconds): # 打印时间的工具
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def train_model_with_tscv(X_train, y_train, model_type='dt', n_splits=5): # 模型训练
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if model_type == 'dt':
        model_class = DecisionTreeClassifier
    elif model_type == 'rf':
        model_class = RandomForestClassifier
    elif model_type == 'xgb':
        model_class = XGBClassifier(base_score=0.5)

    else:
        raise ValueError("Unsupported model type")

    models = []
    scores = []
    times = []

    print(f"Training {model_type.upper()} model with {n_splits}-fold TimeSeriesSplit...\n")
    for i, (train_index, val_index) in enumerate(tqdm(tscv.split(X_train), total=n_splits, desc="Progress")):
        start_time = time.time()

        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = model_class()
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        models.append(model)
        scores.append(score)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        print(f"Fold {i + 1}: Accuracy={score:.4f}, Time={elapsed_time:.2f} seconds")

    best_model = models[np.argmax(scores)]
    print(f"\nBest accuracy: {max(scores):.4f}")
    print(f"Average training time per fold: {np.mean(times):.2f} seconds")
    print(f"total use time : {time.time() - start_time:.2f} seconds")
    return best_model

def select_stocks_and_backtest(model, test_data, hold_data, factor_cols, return_col,
                     imputer, top_k=15, test_start=None, test_end=None, hold_start=None,  target_label=5): # 选股回测
    period_str = f"[test period: {test_start} → {test_end}]"

    test_data = test_data.rename(columns={'code': 'stock_id'})
    hold_data = hold_data.rename(columns={'code': 'stock_id'})

    # 只取测试集结束（hold集开始）那一天的横截面
    test_day_data = test_data[test_data['date'] == test_end]
    for i in (0,10): # 处理Hold集开始非交易日的情况
        if not test_day_data.empty:
            break
        test_day_data = test_data[test_data['date'] == test_end - pd.Timedelta(days=i)]
    if test_day_data.empty:
        print(f"{period_str} ⚠️ test_data 中未找到日期为 {test_start} 的数据，跳过本轮选股。")
        return np.nan, [np.nan] * len(factor_cols)

    # 选股阶段：只用那天的横截面数据
    X_test = test_day_data[factor_cols]
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    if target_label not in model.classes_:
        print(f"{period_str} ⚠️ 当前模型未包含 label={target_label}，模型标签为：{model.classes_}，跳过本轮选股。")
        return np.nan, [np.nan] * len(factor_cols)

    label_index = list(model.classes_).index(target_label)
    y_pred = model.predict_proba(X_test)[:, label_index]

    # 存储预测准确率（新增）
    if 'label' in test_day_data.columns:
        y_true = test_day_data['label'].values
        y_pred_labels = model.predict(X_test)
        accuracy = (y_pred_labels == y_true).mean()
        # 定义保存文件路径
        acc_file = os.path.join(params['result_dir'], f"label_accuracy_{params['model_type']}.csv")
        # 如果文件不存在，先写入表头
        if not os.path.exists(acc_file):
            with open(acc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'accuracy'])
        # 追加本次结果
        with open(acc_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([test_start, f"{accuracy:.4f}"])
        # —— 新增结束 ——

    pred_df = pd.DataFrame({
        'stock_id': test_day_data['stock_id'],
        'score': y_pred
    })

    agg_pred_df = pred_df.groupby('stock_id', as_index=False)['score'].mean()
    top_stocks = agg_pred_df.sort_values(by='score', ascending=False).head(top_k)
    selected_ids = top_stocks['stock_id'].astype(str).tolist()

    top_stocks = agg_pred_df.sort_values(by='score', ascending=False).head(top_k)
    # 从当天的 test_day_data 中，用 stock_id 做索引，提取所有因子列
    features_df = test_day_data.set_index('stock_id')[factor_cols]
    # 把 score 和因子值合并到一个表里
    top_with_features = top_stocks.set_index('stock_id').join(features_df)
    # 重置索引，方便输出
    top_with_features = top_with_features.reset_index()
    # 写入文件时，把所有列都输出
    output_file = os.path.join(params['result_dir'], f"top_k_stocks_{params['model_type']}.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{test_start}  🔎 Top-{top_k} 选股及预测标签及因子值：\n")
        # index=False 去掉行号，columns 自动包含 ['stock_id', 'score'] + factor_cols
        f.write(top_with_features.to_string(index=False))

    hold_returns = hold_data[
        (hold_data['date'] == hold_start) &
        (hold_data['stock_id'].astype(str).isin(selected_ids))
    ]

    error_prefix = f"error_{str(test_start)[:10].replace('-', '')}"
    if hold_returns.empty:
        print(f"{period_str} ⚠️ hold_data 中未找到任何选中股票，跳过该期。")
        top_stocks.to_csv(f"./debug/{error_prefix}_no_hold_data.csv", index=False)
        avg_return = np.nan
    elif hold_returns[return_col].isnull().all():
        print(f"{period_str} ⚠️ 所有选中股票在 hold_data 中 future_return 全为空，股票列表如下：")
        print(top_stocks)
        hold_returns.to_csv(f"./debug/{error_prefix}_all_return_nan.csv", index=False)
        top_stocks.to_csv(f"./debug/{error_prefix}_top_stocks.csv", index=False)
        avg_return = np.nan
    else:
        valid_returns = hold_returns[return_col].dropna()
        if len(valid_returns) < 5:
            print(f"{period_str} ⚠️ 有效收益样本少于5个（仅 {len(valid_returns)} 支），跳过该期。")
            hold_returns.to_csv(f"./debug/{error_prefix}_too_few_valid.csv", index=False)
            avg_return = np.nan
        else:
            avg_return = valid_returns.mean()
            print(f"{period_str} ✅ 成功回测：平均收益为 {avg_return:.4f}，选股数 {len(valid_returns)}")

    return avg_return, model.feature_importances_

# def _run_one_round(
#     round_info,
#     df, factor_cols, label_col, return_col, stock_id_col, stock_list_df
#
# ):
#     """
#     round_info: dict 包含 start_idx, train_days, test_days, hold_days, dates 等
#     其他参数同 backtest_pipeline 的输入
#     返回：(result_dict, feature_importance_dict)
#     """
#     # 解包
#     start_idx = round_info["start_idx"]
#     dates = round_info["dates"]
#     train_days = round_info["train_days"]
#     test_days  = round_info["test_days"]
#     hold_days  = round_info["hold_days"]
#
#     # 1. 计算时间截点
#     train_start = dates[start_idx]
#     train_end   = dates[start_idx + train_days - 1]
#     test_start  = dates[start_idx + train_days]
#     test_end    = dates[start_idx + train_days + test_days - 1]
#     hold_start  = dates[start_idx + train_days + test_days]
#     hold_end    = dates[start_idx + train_days + test_days + hold_days - 1]
#
#     # 2. 筛数据
#     df['date'] = pd.to_datetime(df['date'])
#     train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
#     test_data  = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
#     hold_data  = df[(df['date'] > test_end)   & (df['date'] <= hold_end)]
#
#     # 3. 限制股票池
#     stock_universe = get_stock_list_for_date(test_start, stock_list_df)
#     test_data  = test_data[test_data[stock_id_col].astype(str).isin(stock_universe)]
#     hold_data  = hold_data[hold_data[stock_id_col].astype(str).isin(stock_universe)]
#
#     # 4. 构造训练特征、标签
#     X_train = train_data[factor_cols]
#     y_train = train_data[label_col]
#
#     # 5. 缺失值处理 pipeline
#     inf2nan = FunctionTransformer(
#         func=lambda X: np.where(np.isfinite(X), X, np.nan),
#         validate=False
#     )
#     pipe = Pipeline([
#         ('inf2nan', inf2nan),
#         ('imputer', SimpleImputer(strategy='mean')),
#     ])
#     X_train = pd.DataFrame(
#         pipe.fit_transform(X_train),
#         columns=factor_cols, index=X_train.index
#     )
#     # 同理填充 test & hold
#     test_data_filled = test_data.copy()
#     test_data_filled[factor_cols] = pipe.transform(test_data[factor_cols])
#     hold_data_filled = hold_data.copy()
#     hold_data_filled[factor_cols] = pipe.transform(hold_data[factor_cols])
#
#     # 6. 训练模型
#     model = train_model_with_tscv(X_train, y_train, model_type=params['model_type'])
#
#     # 7. 选股并回测
#     avg_return, feat_importance = select_stocks_and_backtest(
#         model=model,
#         test_data=test_data_filled, hold_data=hold_data_filled,
#         factor_cols=factor_cols, return_col=return_col,
#         imputer=pipe.named_steps['imputer'],
#         top_k=15,
#         test_start=test_start,
#         test_end=test_end,
#         hold_start=hold_start
#     )
#
#     # 8. 组织结果
#     result = {
#         'test_period_start': test_start,
#         'test_period_end': hold_end,
#         'avg_return': avg_return
#     }
#     feat_imp_dict = {'date': test_start}
#     feat_imp_dict.update({f: v for f, v in zip(factor_cols, feat_importance)})
#
#     return result, feat_imp_dict
#
#
# def backtest_pipeline(
#     df, factor_cols, label_col, return_col, stock_id_col,
#     stock_list_df,
#     train_years=3, test_years=1, hold_months=4, step_months=4,
#     n_jobs=4
# ):
#     """
#     并行版回测主函数
#     n_jobs: 并发线程数
#     """
#     # --- 预处理，计算各轮参数 ---
#     stock_list_df['date'] = pd.to_datetime(stock_list_df['date'])
#     valid_start_date = stock_list_df[stock_list_df['stock_count'] > 0]['date'].min()
#     df['date'] = pd.to_datetime(df['date'])
#     dates = sorted(df['date'].unique())
#     # 起始索引
#     start_idx = next((i for i,d in enumerate(dates) if d >= valid_start_date), 0)
#
#     train_days = train_years * 252
#     test_days  = test_years  * 252
#     hold_days  = hold_months * 21
#     step_days  = step_months * 21
#
#     # 构造每轮参数列表
#     rounds = []
#     idx = start_idx
#     while idx + train_days + test_days + hold_days <= len(dates):
#         rounds.append({
#             "start_idx": idx,
#             "dates": dates,
#             "train_days": train_days,
#             "test_days": test_days,
#             "hold_days": hold_days
#         })
#         idx += step_days
#
#     # --- 并行执行 ---
#     results = []
#     feature_importance = []
#     with ProcessPoolExecutor(max_workers=16) as exe:
#         futures = [
#             exe.submit(
#                 _run_one_round,
#                 info, df, factor_cols, label_col, return_col,
#                 stock_id_col, stock_list_df
#             )
#             for info in rounds
#         ]
#         for fut in as_completed(futures):
#             res, feat_imp = fut.result()
#             results.append(res)
#             feature_importance.append(feat_imp)
#
#     # 按日期排序（可选）
#     results_df = pd.DataFrame(results).sort_values('test_period_start').reset_index(drop=True)
#     fi_df      = pd.DataFrame(feature_importance).sort_values('date').reset_index(drop=True)
#
#     return results_df, fi_df


def backtest_pipeline(
    df: pd.DataFrame,
    factor_cols: list,
    label_col: str,
    return_col: str,
    stock_id_col: str,
    stock_list_df: pd.DataFrame,
    schedule_csv: str = "./data/processed/zz500_list_filter.csv",
    top_k: int = 15,
    model_type: str = "dt",
):
    """基于中证 500 调仓节奏的回测框架。

    参数说明
    ----------
    df : pd.DataFrame
        主因子+标签数据，必须包含 ``date`` 列。
    factor_cols : list[str]
        因子列名列表。
    label_col : str
        监督学习标签列名。
    return_col : str
        个股未来持仓期收益列名，用于回测。
    stock_id_col : str
        股票代码列名。
    stock_list_df : pd.DataFrame
        股票池 DataFrame，含 ``date`` 与 ``stock_count``，使用 ``get_stock_list_for_date`` 做过滤。
    schedule_csv : str, default "./data/processed/zz500_list_filter.csv"
        调仓窗口定义文件，须含 ``train_date``, ``test_date``, ``buy_date``, ``end_date`` 四列。
    top_k : int, default 15
        每轮选出买入的股票数量。
    model_type : str, default "lgbm"
        传递给 ``train_model_with_tscv`` 的模型类型。
    """

    # ‼️ 1. 载入调仓日表并排序
    schedule_df = pd.read_csv(
        schedule_csv,
        parse_dates=["train_date", "test_date", "buy_date", "end_date"],
    ).sort_values("train_date").reset_index(drop=True)

    # 2. 日期列标准化
    df["date"] = pd.to_datetime(df["date"])
    stock_list_df["date"] = pd.to_datetime(stock_list_df["date"])

    results = []
    feature_importance_list = []

    total_rounds = len(schedule_df)
    start_time = time.time()
    for ridx, row in schedule_df.iterrows():
        # 用于测试 TEST
        TEST_FLAG = False # 测试时改为True
        if TEST_FLAG and ridx > 5 : # 还可以改为其他条件
            break
        # 测试段结束
        # === 2.1 读取时间边界 ===
        train_start = row["train_date"]
        test_start = row["test_date"]
        hold_start = row["buy_date"]  # 中证 500 阶段性调仓后买入时间
        hold_end = row["end_date"]
        train_end = test_start - pd.Timedelta(days=1)
        test_end = hold_start - pd.Timedelta(days=1)

        # 如数据不足，则跳过
        if hold_end > df["date"].max():
            print(
                f"⚠️  数据不足，跳过第 {ridx + 1} 轮（hold_end={hold_end.date()} 超出数据范围）"
            )
            continue

        print(
            f"\n================ 开始回测第 {ridx + 1} / {total_rounds} 轮 ================\n"
        )
        print(
            f"训练集: {train_start.date()} ➜ {train_end.date()} | "
            f"测试集: {test_start.date()} ➜ {test_end.date()} | "
            f"持仓期: {hold_start.date()} ➜ {hold_end.date()}"
        )

        # === 2.2 切片数据 ===
        train_data = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        hold_data = df[(df["date"] >= hold_start) & (df["date"] <= hold_end)]

        # === 2.3 股票池过滤 ===
        stock_universe = get_stock_list_for_date(hold_start, stock_list_df)
        test_data = test_data[test_data[stock_id_col].astype(str).isin(stock_universe)]
        hold_data = hold_data[hold_data[stock_id_col].astype(str).isin(stock_universe)]
        # 股票池过滤后立即打印股票数量
        print(f"📊 当前测试集股票数量：{test_data[stock_id_col].nunique()} 只")
        # === 2.4 特征 & 标签 ===
        X_train = train_data[factor_cols]
        y_train = train_data[label_col]
        print("✅ 本轮训练标签种类：", sorted(y_train.unique()))

        # === 2.5 缺失值处理 ===
        inf2nan = FunctionTransformer(lambda X: np.where(np.isfinite(X), X, np.nan), validate=False)
        pre_pipe = Pipeline([
            ("inf2nan", inf2nan),
            ("imputer", SimpleImputer(strategy="mean")),
        ])

        X_train = pd.DataFrame(pre_pipe.fit_transform(X_train), columns=factor_cols, index=X_train.index)
        test_data_filled = test_data.copy()
        test_data_filled[factor_cols] = pd.DataFrame(
            pre_pipe.transform(test_data[factor_cols]), columns=factor_cols, index=test_data.index
        )
        hold_data_filled = hold_data.copy()
        hold_data_filled[factor_cols] = pd.DataFrame(
            pre_pipe.transform(hold_data[factor_cols]), columns=factor_cols, index=hold_data.index
        )

        # === 2.6 模型训练 ===
        model = train_model_with_tscv(X_train, y_train, model_type=model_type)

        # === 2.7 选股 + 回测 ===
        avg_return, feat_importance = select_stocks_and_backtest(
            model=model,
            test_data=test_data_filled,
            hold_data=hold_data_filled,
            factor_cols=factor_cols,
            return_col=return_col,
            imputer=pre_pipe.named_steps["imputer"],
            top_k=top_k,
            test_start=test_start,
            test_end=test_end,
            hold_start=hold_start,
        )

        results.append({
            "test_period_start": test_start,
            "test_period_end": hold_end,
            "avg_return": avg_return,
        })

        feature_importance_list.append({
            "date": test_start,
            **{factor: val for factor, val in zip(factor_cols, feat_importance)},
        })

        time_cnt = time.time() - start_time
        # 在每轮开始时

        print(
            f"开始时间: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}, "
            f"当前时间: {datetime.now().strftime('%H:%M:%S')}, "
            f"总用时: {format_seconds(time_cnt)}, "
            f"平均每轮用时: {format_seconds(time_cnt / (ridx + 1))}, "
            f"预计剩余用时: {format_seconds(time_cnt / (ridx + 1) * (total_rounds - ridx - 1))}"
        )
        print(f"📊 当前测试集股票数量：{test_data[stock_id_col].nunique()} 只")
    return pd.DataFrame(results), pd.DataFrame(feature_importance_list)



def compute_performance_metrics(backtest_df, risk_free_rate=0.0):
    backtest_df = backtest_df.copy()
    returns = backtest_df['avg_return'].dropna()

    if returns.empty or len(returns) < 2:
        return {
            'Annualized Return': np.nan,
            'Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Max Drawdown': np.nan
        }

    # 计算两个 rebalancing 日期之间的天数
    period_len = (backtest_df['test_period_start'].iloc[1] - backtest_df['test_period_start'].iloc[0]).days
    annual_factor = 252 / period_len if period_len > 0 else 1

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


import matplotlib.pyplot as plt

def plot_cumulative_return(backtest_df, risk_free_rate=0.0):
    """
    绘制累计收益曲线，并以文本框形式展示总体绩效指标：
      - 年化收益 (Annualized Return)
      - 年化波动率 (Volatility)
      - 夏普比率 (Sharpe Ratio)
      - 最大回撤 (Max Drawdown)
    backtest_df 要包含 ['test_period_start', 'avg_return'] 两列。
    """
    # 复制数据，计算累计收益
    df = backtest_df.copy()
    df['cum_return'] = (1 + df['avg_return']).cumprod()

    # 计算总体绩效指标
    metrics = compute_performance_metrics(df, risk_free_rate)

    # 1) 累计收益曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['test_period_start'], df['cum_return'],
             label='Cumulative Return', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # 2) 文本框展示总体绩效指标
    textstr = '\n'.join([
        f"Annualized Return: {metrics['Annualized Return']:.2%}",
        f"Volatility:         {metrics['Volatility']:.2%}",
        f"Sharpe Ratio:       {metrics['Sharpe Ratio']:.2f}",
        f"Max Drawdown:       {metrics['Max Drawdown']:.2%}"
    ])

    plt.figure(figsize=(6, 3))
    plt.axis('off')  # 不显示坐标轴
    # 在图中添加文本
    plt.text(0.01, 0.5, textstr, fontsize=12, va='center')
    plt.title('Performance Metrics')
    plt.tight_layout()
    plt.show()



def plot_full_backtest_performance(backtest_df, risk_free_rate=0.0, show_rf_line=True):
    backtest_df = backtest_df.copy()

    # 策略累计收益
    backtest_df['cum_return'] = (1 + backtest_df['avg_return']).cumprod()

    # 基准累计收益
    if 'benchmark_return' in backtest_df.columns:
        backtest_df['cum_benchmark'] = (1 + backtest_df['benchmark_return']).cumprod()
    else:
        raise ValueError("DataFrame must include 'benchmark_return' column.")

    # 最大回撤计算
    cum_returns = backtest_df['cum_return']
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak

    # 绩效指标
    metrics = compute_performance_metrics(backtest_df, risk_free_rate)

    # 画图
    plt.figure(figsize=(12, 7))

    # 策略累计收益
    plt.plot(backtest_df['test_period_start'], backtest_df['cum_return'], label='Strategy', color='blue',
             linewidth=2)

    # 基准累计收益
    plt.plot(backtest_df['test_period_start'], backtest_df['cum_benchmark'], label='Benchmark', color='gray',
             linestyle='--')

    # 最大回撤阴影
    plt.fill_between(backtest_df['test_period_start'], cum_returns, peak, where=(cum_returns < peak), color='red',
                     alpha=0.2, label='Drawdown')

    # 每个点标注收益
    for i in range(len(backtest_df)):
        x = backtest_df['test_period_start'].iloc[i]
        y = backtest_df['cum_return'].iloc[i]
        plt.scatter(x, y, color='blue', s=30)
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)

    # 无风险参考线（复利年化）
    if show_rf_line:
        n_periods = len(backtest_df)
        period_len = (backtest_df['test_period_start'].iloc[1] - backtest_df['test_period_start'].iloc[0]).days
        annual_factor = 252 / period_len
        rf_per_period = (1 + risk_free_rate) ** (1 / annual_factor) - 1
        backtest_df['cum_rf'] = (1 + rf_per_period) ** np.arange(n_periods)
        plt.plot(backtest_df['test_period_start'], backtest_df['cum_rf'], label='Risk-Free', color='green',
                 linestyle=':')

    # 图例与轴
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Performance with Benchmark & Drawdown')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # 性能指标文本框
    textstr = '\n'.join([
        f"Annualized Return: {metrics['Annualized Return']:.2%}",
        f"Volatility: {metrics['Volatility']:.2%}",
        f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}",
        f"Max Drawdown: {metrics['Max Drawdown']:.2%}"
    ])
    plt.gcf().text(0.15, 0.75, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def get_stock_list_for_date(current_date, stock_list_df):
    # 过滤出所有不晚于当前日期的记录
    eligible_rows = stock_list_df[stock_list_df['date'] <= current_date]
    if eligible_rows.empty:
        return set()

    latest_row = eligible_rows.sort_values('date', ascending=False).iloc[0]
    if pd.isna(latest_row['stock_list']) or latest_row['stock_list'] == '':
        return set()
    return set(str(latest_row['stock_list']).split(','))


if __name__ == '__main__':
    # 1.定义列名
    factor_cols = ['6m_return', '11m_return',
                    '总市值',  # 日度数据
                   'pe', 'pb', 'ps', '现金流比股价',  # 日度季度组合数据
                   '净资产收益率A', '每股收益',  # 季度数据
                   '资本支出比总市值', '流动比率', 'ocfp', 'capex', 'evebit', 'evebitda', '企业价值不含货币资金',
                   '12m_lagged_return', '24m_lagged_return',
                   'Beta3Y_Cov', 'Beta3Y_Reg']
    """
        资本支出 / 总市值
        流动比率： 流动资产（缺）/流动负债（有）
        ocfp: 经营活动现金流量净额(有）/ 净资产（无？）
        capex: 资本支出/营业收入（都有）
        evebit: 企业价值（用哪个？）/ EBIT(季度）
        evebitda: 企业价值（用哪个？）/ EBITDA(季度）
        两个lag，两个回归
    """
    label_col = 'label'  # 分类标签：高/中/低收益（分类问题）
    return_col = 'ret_fwd_4m'  # 实际未来收益率（连续值，用于回测）
    stock_id_col = 'code'  # 股票代码 后面rename
    cols_input = ['6m_return', '11m_return', '总市值',  # 日度数据
                   'pe', 'pb', 'ps', '现金流比股价',  # 日度季度组合数据
                   '净资产收益率A', '每股收益',  # 季度数据
                   '资本支出比总市值', '流动比率', 'ocfp', 'capex', 'evebit', 'evebitda', '企业价值不含货币资金',
                   '12m_lagged_return', '24m_lagged_return',
                   'Beta3Y_Cov', 'Beta3Y_Reg',
                   'date', 'code', 'label', 'ret_fwd_4m']

    # 2. 读取并准备数据
    # df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), encoding='utf-8-sig', usecols=cols_input)
    df = data_loader.get_daily_price_ret_pd(usecols=cols_input)
    df['date'] = pd.to_datetime(df['date'])  # 确保日期列为 datetime 类型
    df.sort_values(['date', 'code'], inplace=True)
    # df.rename(columns={'code': 'stock_id', 'Beta3Y_Cov_y': 'Beta3Y_Cov', 'Beta3Y_Reg_y':'Beta3Y_Reg'}, inplace=True)

    # 读取股票池数据
    # stock_list_df = pd.read_csv(os.path.join(params['data_dir'], './best_stock_window_snapshot.csv'), parse_dates=['date'])
    stock_list_df = data_loader.get_stock_list_pd()

    # 清空输出文档

    la_file = os.path.join(params['result_dir'], f"top_k_stocks_{params['model_type']}.txt")
    with open(la_file, "w", encoding="utf-8") as f:
        pass

    # 3. 执行回测流程
    backtest_df, feature_df = backtest_pipeline(
        df=df,
        factor_cols=factor_cols,
        label_col=label_col,
        return_col=return_col,
        stock_id_col=stock_id_col,
        stock_list_df=stock_list_df,
        # train_years=3,
        # test_years=1,
        # hold_months=4,
        # step_months=1000,  # 测试时改大一点，算的快，基准为4
    )

    # 4. 输出回测结果与因子重要性
    backtest_df.to_csv(os.path.join(params['result_dir'], f"backtest_results_{params['model_type']}.csv"), index=False)
    feature_df.to_csv(os.path.join(params['result_dir'], f"feature_importance_time_series_{params['model_type']}.csv"),
                      index=False)

    # 5. 可视化收益曲线
    plot_cumulative_return(backtest_df)
    print("plot finish")
    # plot_full_backtest_performance(backtest_df, risk_free_rate=0.0, show_rf_line=False)
    draw.draw_all()
    print("回测完成，结果已保存！")




# def clean_nan(X, strategy='mean'):  # 废弃
#     if strategy not in ['mean', 'median']:
#         raise ValueError("strategy must be 'mean' or 'median'")
#
#     imputer = SimpleImputer(strategy=strategy)
#     X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
#
#     return X_clean, imputer


# def select_stocks_and_backtest2(model, test_data, hold_data, factor_cols, return_col,
#                                imputer, top_k=15, test_start=None, test_end=None, hold_start=None):
#     period_str = f"[test period: {test_start} → {test_end}]"
#
#     # 1. 只取测试区间的数据
#     mask = (test_data['date'] >= test_start) & (test_data['date'] <= test_end)
#     test_window = test_data.loc[mask]
#     if test_window.empty:
#         print(f"{period_str} ⚠️ 在 test_data 中未找到 {test_start} 到 {test_end} 的数据，跳过本轮选股。")
#         return np.nan, [np.nan] * len(factor_cols)
#
#     # 2. 对每个测试日单独预测标签，并收集
#     preds_list = []
#     for day, grp in test_window.groupby('date'):
#         X_day = grp[factor_cols]
#         X_day = pd.DataFrame(imputer.transform(X_day),
#                              columns=factor_cols, index=grp.index)
#         # 这里使用 predict 返回离散标签
#         y_day_pred = model.predict(X_day)
#         df_day = pd.DataFrame({
#             'stock_id': grp['stock_id'].astype(str),
#             'pred_label': y_day_pred
#         }, index=grp.index)
#         preds_list.append(df_day)
#
#     all_preds = pd.concat(preds_list)
#
#     # 3. 聚合：按 stock_id 取平均预测标签
#     agg_pred_df = all_preds.groupby('stock_id', as_index=False)['pred_label'].mean()
#     # 4. 根据平均标签排序，选 top_k
#     top_stocks = agg_pred_df.sort_values(by='pred_label', ascending=False).head(top_k)
#     selected_ids = top_stocks['stock_id'].tolist()
#
#     # 将选股结果追加保存
#     output_file = os.path.join(params['result_dir'], f"top_k_stocks_{params['model_type']}.txt")
#     with open(output_file, "a", encoding="utf-8") as f:
#         f.write("\n")
#         f.write(f"{test_start} → {test_end}  🔎 Top-{top_k} 选股（平均预测标签）：\n")
#         f.write(top_stocks.to_string(index=False))
#         f.write("\n\n")
#
#     # 5. 回测：在 hold_data 中取 hold_start 当日的表现
#     hold_returns = hold_data[
#         (hold_data['date'] == hold_start) &
#         (hold_data['stock_id'].astype(str).isin(selected_ids))
#         ]
#
#     error_prefix = f"error_{str(test_start)[:10].replace('-', '')}"
#     if hold_returns.empty:
#         print(f"{period_str} ⚠️ hold_data 中未找到任何选中股票，跳过该期。")
#         avg_return = np.nan
#     elif hold_returns[return_col].isnull().all():
#         print(f"{period_str} ⚠️ 所有选中股票在 hold_data 中 {return_col} 全为空，跳过该期。")
#         avg_return = np.nan
#     else:
#         valid_returns = hold_returns[return_col].dropna()
#         if len(valid_returns) < 5:
#             print(f"{period_str} ⚠️ 有效收益样本少于5个（仅 {len(valid_returns)} 支），跳过该期。")
#             avg_return = np.nan
#         else:
#             avg_return = valid_returns.mean()
#             print(f"{period_str} ✅ 成功回测：平均收益为 {avg_return:.4f}，选股数 {len(valid_returns)}")
#
#     # 返回平均收益和原模型的特征重要性（如有）
#     fi = getattr(model, 'feature_importances_', [np.nan] * len(factor_cols))
#     return avg_return, fi