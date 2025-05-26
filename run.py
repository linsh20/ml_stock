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
from src import data_loader
from datetime import timedelta
from joblib import Parallel, delayed

MODEL_TYPE = 'dt'


def format_seconds(seconds): # 打印时间的工具
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def train_model_with_tscv(X_train, y_train, model_type='dt', n_splits=5, random_seed=29, criterion = 'gini',
                          n_jobs = 1): # 模型训练
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if model_type == 'dt':
        model_class = DecisionTreeClassifier
        model_kwargs = {'random_state': random_seed, 'criterion': criterion}
    elif model_type == 'rf':
        model_class = RandomForestClassifier
        model_kwargs = {'n_jobs': n_jobs, 'random_state': random_seed}
    elif model_type == 'xgb':
        model_class = XGBClassifier
        model_kwargs = {
            'base_score': 0.5,
            'tree_method': 'hist',
            'n_jobs': n_jobs,
            'random_state': random_seed,
            'verbosity': 0  # 可选：避免多线程打印冲突
        }
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

        model = model_class(**model_kwargs)
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
    test_day_data = test_data[test_data['date'] == test_end] # TODO
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
        f.write(top_with_features.to_string(index=False))

    # 如果hold_start不是交易日，则后移Hold_start，直到遇到交易日为止
    attempts = 0
    while attempts < 10:
        hold_returns = hold_data[
            (hold_data['date'] == hold_start) &
            (hold_data['stock_id'].astype(str).isin(selected_ids))
            ]
        if not hold_returns.empty:
            break  # 找到了非空结果，退出循环
        # 否则后移一天
        hold_start += timedelta(days=1)
        attempts += 1
    # 如果尝试了10次都没有找到非空结果，可以额外加一个提示或处理逻辑
    if hold_returns.empty:
        print(
            f"Warning: No holding returns found after {10} days starting from {hold_start - timedelta(days=10)}.")

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


def run_single_backtest(
    ridx,
    row,
    df,
    factor_cols,
    label_col,
    return_col,
    stock_id_col,
    top_k,
    model_type,
    total_rounds,
):
    """
    对单次调仓回测逻辑进行封装，供 joblib 并行调用。
    """
    train_start = row["train_date"]
    test_start = row["test_date"]
    hold_start = row["buy_date"]
    hold_end = row["end_date"]
    train_end = test_start - pd.Timedelta(days=1)
    test_end = hold_start - pd.Timedelta(days=1)

    # 数据不足时跳过
    if hold_end > df["date"].max():
        print(
            f"⚠️ 数据不足，跳过第 {ridx + 1} 轮（hold_end={hold_end.date()} 超出数据范围）"
        )
        return None

    print(f"\n========== 回测第 {ridx + 1} / {total_rounds} 轮 ==========")
    print(
        f"训练集: {train_start.date()} ➜ {train_end.date()} | "
        f"测试集: {test_start.date()} ➜ {test_end.date()} | "
        f"持仓期: {hold_start.date()} ➜ {hold_end.date()}"
    )

    # 切片数据
    train_data = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    hold_data = df[(df["date"] >= hold_start) & (df["date"] <= hold_end)]

    # 股票池过滤
    stock_universe = set(str(row["stock_list"]).split(","))
    test_data = test_data[test_data[stock_id_col].astype(str).isin(stock_universe)]
    hold_data = hold_data[hold_data[stock_id_col].astype(str).isin(stock_universe)]
    print(f"📊 本轮测试集股票数量：{test_data[stock_id_col].nunique()} 只")

    # 构建特征与标签
    X_train = train_data[factor_cols]
    y_train = train_data[label_col]
    print("✅ 本轮训练标签种类：", sorted(y_train.unique()))

    # 缺失值处理流水线
    inf2nan = FunctionTransformer(
        lambda X: np.where(np.isfinite(X), X, np.nan), validate=False
    )
    pre_pipe = Pipeline(
        [
            ("inf2nan", inf2nan),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )

    X_train_filled = pd.DataFrame(
        pre_pipe.fit_transform(X_train), columns=factor_cols, index=X_train.index
    )

    test_data_filled = test_data.copy()
    test_data_filled[factor_cols] = pd.DataFrame(
        pre_pipe.transform(test_data[factor_cols]),
        columns=factor_cols,
        index=test_data.index,
    )

    hold_data_filled = hold_data.copy()
    hold_data_filled[factor_cols] = pd.DataFrame(
        pre_pipe.transform(hold_data[factor_cols]),
        columns=factor_cols,
        index=hold_data.index,
    )

    # 模型训练
    model = train_model_with_tscv(X_train_filled, y_train, model_type=model_type)

    # 选股与回测
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

    return {
        "result": {
            "test_period_start": test_start,
            "test_period_end": hold_end,
            "avg_return": avg_return,
        },
        "importance": {
            "date": test_start,
            **{factor: val for factor, val in zip(factor_cols, feat_importance)},
        },
    }


def backtest_pipeline(
    df: pd.DataFrame,
    factor_cols: list,
    label_col: str,
    return_col: str,
    stock_id_col: str,
    schedule_csv: str = "./data/processed/zz500_list_filter.csv",
    top_k: int = 15,
    model_type: str = MODEL_TYPE,
):
    """
    基于中证 500 调仓节奏的回测框架——并行版（使用 joblib 多线程）。
    """
    # 1. 载入调仓日表并排序
    schedule_df = (
        pd.read_csv(
            schedule_csv,
            parse_dates=["train_date", "test_date", "buy_date", "end_date"],
        )
        .sort_values("train_date")
        .reset_index(drop=True)
    )

    # 2. 日期列标准化
    df["date"] = pd.to_datetime(df["date"])

    total_rounds = len(schedule_df)
    start_time = time.time()

    # 测试模式：仅跑前 N 轮 TODO
    TEST_FLAG = False
    max_round = 5 if TEST_FLAG else total_rounds

    # 并行执行每轮回测
    results_all = Parallel(n_jobs=8)(
        delayed(run_single_backtest)(
            ridx,
            row,
            df,
            factor_cols,
            label_col,
            return_col,
            stock_id_col,
            top_k,
            model_type,
            total_rounds,
        )
        for ridx, row in schedule_df.iterrows()
        if ridx < max_round
    )

    # 过滤 None
    results = [r["result"] for r in results_all if r is not None]
    feature_importance_list = [r["importance"] for r in results_all if r is not None]

    # 总用时打印
    time_cnt = time.time() - start_time
    print(f"🚀 回测完成（{len(results)} 轮），总用时：{format_seconds(time_cnt)}")

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


if __name__ == '__main__':
    os.makedirs('./debug', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
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
