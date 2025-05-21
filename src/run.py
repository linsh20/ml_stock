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
from tqdm import tqdm
import draw
import csv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

def train_model_with_tscv(X_train, y_train, model_type='dt', n_splits=5):
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


def select_stocks_and_backtest2(model, test_data, hold_data, factor_cols, return_col,
                               imputer, top_k=15, test_start=None, test_end=None, hold_start=None):
    period_str = f"[test period: {test_start} → {test_end}]"

    # 1. 只取测试区间的数据
    mask = (test_data['date'] >= test_start) & (test_data['date'] <= test_end)
    test_window = test_data.loc[mask]
    if test_window.empty:
        print(f"{period_str} ⚠️ 在 test_data 中未找到 {test_start} 到 {test_end} 的数据，跳过本轮选股。")
        return np.nan, [np.nan] * len(factor_cols)

    # 2. 对每个测试日单独预测标签，并收集
    preds_list = []
    for day, grp in test_window.groupby('date'):
        X_day = grp[factor_cols]
        X_day = pd.DataFrame(imputer.transform(X_day),
                             columns=factor_cols, index=grp.index)
        # 这里使用 predict 返回离散标签
        y_day_pred = model.predict(X_day)
        df_day = pd.DataFrame({
            'stock_id': grp['stock_id'].astype(str),
            'pred_label': y_day_pred
        }, index=grp.index)
        preds_list.append(df_day)

    all_preds = pd.concat(preds_list)

    # 3. 聚合：按 stock_id 取平均预测标签
    agg_pred_df = all_preds.groupby('stock_id', as_index=False)['pred_label'].mean()
    # 4. 根据平均标签排序，选 top_k
    top_stocks = agg_pred_df.sort_values(by='pred_label', ascending=False).head(top_k)
    selected_ids = top_stocks['stock_id'].tolist()

    # 将选股结果追加保存
    output_file = os.path.join(params['result_dir'], f"top_k_stocks_{params['model_type']}.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{test_start} → {test_end}  🔎 Top-{top_k} 选股（平均预测标签）：\n")
        f.write(top_stocks.to_string(index=False))
        f.write("\n\n")

    # 5. 回测：在 hold_data 中取 hold_start 当日的表现
    hold_returns = hold_data[
        (hold_data['date'] == hold_start) &
        (hold_data['stock_id'].astype(str).isin(selected_ids))
        ]

    error_prefix = f"error_{str(test_start)[:10].replace('-', '')}"
    if hold_returns.empty:
        print(f"{period_str} ⚠️ hold_data 中未找到任何选中股票，跳过该期。")
        avg_return = np.nan
    elif hold_returns[return_col].isnull().all():
        print(f"{period_str} ⚠️ 所有选中股票在 hold_data 中 {return_col} 全为空，跳过该期。")
        avg_return = np.nan
    else:
        valid_returns = hold_returns[return_col].dropna()
        if len(valid_returns) < 5:
            print(f"{period_str} ⚠️ 有效收益样本少于5个（仅 {len(valid_returns)} 支），跳过该期。")
            avg_return = np.nan
        else:
            avg_return = valid_returns.mean()
            print(f"{period_str} ✅ 成功回测：平均收益为 {avg_return:.4f}，选股数 {len(valid_returns)}")

    # 返回平均收益和原模型的特征重要性（如有）
    fi = getattr(model, 'feature_importances_', [np.nan] * len(factor_cols))
    return avg_return, fi




def select_stocks_and_backtest(model, test_data, hold_data, factor_cols, return_col,
                     imputer, top_k=15, test_start=None, test_end=None, hold_start=None,  target_label=5):
    period_str = f"[test period: {test_start} → {test_end}]"

    # 只取测试集开始那一天的横截面
    test_day_data = test_data[test_data['date'] == test_end]
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


def backtest_pipeline(df, factor_cols, label_col, return_col, stock_id_col,  stock_list_df,  # 主函数
                      train_years=3, test_years=1, hold_months=4, step_months=4):
    results = []
    feature_importance_list = []

    # 找到第一个 stock_count > 0 的日期
    stock_list_df['date'] = pd.to_datetime(stock_list_df['date'])
    valid_start_date = stock_list_df[stock_list_df['stock_count'] > 0]['date'].min()

    # 所有交易日期按时间顺序排序
    df['date'] = pd.to_datetime(df['date'])
    dates = sorted(df['date'].unique())

    # 找到 valid_start_date 在 df 中的索引，作为 start_idx
    start_idx = next((i for i, d in enumerate(dates) if d >= valid_start_date), 0)

    # 定义每次回测长度（以交易日为单位）
    train_days = train_years * 252
    test_days = test_years * 252
    step_days = step_months * 21
    hold_days = hold_months * 21

    start_idx_cnt, round_cnt = start_idx, 0
    # 计算一下一共多少轮，方便打印
    while start_idx_cnt + train_days + test_days + hold_days <= len(dates):
        start_idx_cnt += step_days
        round_cnt += 1

    round_num = 1
    while start_idx + train_days + test_days + hold_days <= len(dates):
        print(f"\n================ 开始回测第 {round_num} / {round_cnt} 轮 ================\n")
        # 计算时间范围
        train_start = dates[start_idx] # 3年
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]  # 1年
        test_end = dates[start_idx + train_days + test_days - 1]
        hold_start = dates[start_idx + train_days + test_days]
        hold_end = dates[start_idx + train_days + test_days + hold_days - 1] # 4个月

        train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
        test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
        hold_data = df[(df['date'] > test_end) & (df['date'] <= hold_end)]

        # test 和 hold 做限制
        stock_universe = get_stock_list_for_date(test_start, stock_list_df)  # 从stock_list_df 获取对应日期的股票列表
        test_data = test_data[test_data[stock_id_col].astype(str).isin(stock_universe)]  # 取交集
        hold_data = hold_data[hold_data[stock_id_col].astype(str).isin(stock_universe)]

        # 模型训练
        X_train = train_data[factor_cols]
        y_train = train_data[label_col]
        print("✅ 本轮训练标签种类：", sorted(y_train.unique()))
        # ==========更robust的缺失值处理===============
        inf2nan = FunctionTransformer(
            func=lambda X: np.where(np.isfinite(X), X, np.nan),
            validate=False
        )

        pipeline = Pipeline([
            ('inf2nan', inf2nan),
            ('imputer', SimpleImputer(strategy='mean')),
        ])

        X_train = pd.DataFrame(
            pipeline.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        test_data_filled = test_data.copy()
        test_data_filled[factor_cols] = pd.DataFrame(
            pipeline.transform(test_data[factor_cols]),
            columns=factor_cols,
            index=test_data.index
        )

        hold_data_filled = hold_data.copy()
        hold_data_filled[factor_cols] = pd.DataFrame(
            pipeline.transform(hold_data[factor_cols]),
            columns=factor_cols,
            index=hold_data.index
        )
        # 缺失值处理->生成impute
        X_all_for_impute = X_train.copy()
        # X_all_for_impute.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(X_all_for_impute.isna().sum()[lambda x: x > 0])
        imputer_test = SimpleImputer(strategy='mean')
        imputer_test.fit(X_all_for_impute)
        # 缺失值处理->使用impute
        X_train = pd.DataFrame(imputer_test.transform(X_train), columns=X_train.columns, index=X_train.index)
        test_data_filled = test_data.copy()
        test_data_filled[factor_cols] = pd.DataFrame(
            imputer_test.transform(test_data[factor_cols]),
            columns=factor_cols,
            index=test_data.index
        )
        hold_data_filled = hold_data.copy()
        hold_data_filled[factor_cols] = pd.DataFrame(
            imputer_test.transform(hold_data[factor_cols]),
            columns=factor_cols,
            index=hold_data.index
        )

        # 训练模型
        model = train_model_with_tscv(X_train, y_train, model_type=params['model_type'])
        # 选股 + 回测收益
        avg_return, feat_importance = select_stocks_and_backtest(
            model=model,
            test_data=test_data_filled,
            hold_data=hold_data_filled,
            factor_cols=factor_cols,
            return_col=return_col,
            imputer=imputer_test,
            top_k=15,
            test_start=test_start,
            test_end=test_end,
            hold_start=hold_start
        )

        results.append({
            'test_period_start': test_start,
            'test_period_end': hold_end,
            'avg_return': avg_return
        })

        feature_importance_list.append({
            'date': test_start,
            **{factor: val for factor, val in zip(factor_cols, feat_importance)}
        })

        start_idx += step_days
        round_num += 1

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
    stock_id_col = 'stock_id'  # 股票代码 后面rename
    cols_input = ['6m_return', '11m_return', '总市值',  # 日度数据
                   'pe', 'pb', 'ps', '现金流比股价',  # 日度季度组合数据
                   '净资产收益率A', '每股收益',  # 季度数据
                   '资本支出比总市值', '流动比率', 'ocfp', 'capex', 'evebit', 'evebitda', '企业价值不含货币资金',
                   '12m_lagged_return', '24m_lagged_return',
                   'Beta3Y_Cov_y', 'Beta3Y_Reg_y',
                   'date', 'code', 'label', 'ret_fwd_4m']

    # 2. 读取并准备数据
    df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), encoding='utf-8-sig', usecols=cols_input)
    df['date'] = pd.to_datetime(df['date'])  # 确保日期列为 datetime 类型
    df.sort_values(['date', 'code'], inplace=True)
    df.rename(columns={'code': 'stock_id', 'Beta3Y_Cov_y': 'Beta3Y_Cov', 'Beta3Y_Reg_y':'Beta3Y_Reg'}, inplace=True)

    # 读取股票池数据
    stock_list_df = pd.read_csv(os.path.join(params['data_dir'], './best_stock_window_snapshot.csv'), parse_dates=['date'])

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
        train_years=3,
        test_years=1,
        hold_months=4,
        step_months=1000,  # 测试时改大一点，算的快，基准为4
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