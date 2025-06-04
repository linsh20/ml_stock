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
import logging
from logging.handlers import TimedRotatingFileHandler
import requests
import src.data_loader as dl
import argparse
from sklearn.preprocessing import LabelEncoder


def parse_args():
    # 解析参数
    parser = argparse.ArgumentParser(description="Run time series model.")
    parser.add_argument('--model_type', type=str, default='dt', choices=['dt', 'rf', 'xgb'],
                        help="Type of model to use: 'dt' (Decision Tree), 'rf' (Random Forest), 'xgb' (XGBoost)")
    parser.add_argument('--test', action='store_true', help="Whether to run in test mode")
    parser.add_argument('--n_jobs', type=int, default=8,
                        help="Number of parallel jobs to run. -1 means using all processors.")
    parser.add_argument('--top_k', type=int, default=15,
                        help="Number of top stocks to select based on predicted scores.")
    parser.add_argument('--k_fold', type=int, default=5,
                        help="Number of folds in time-series cross-validation.")
    args = parser.parse_args()
    return args


def init_logs():
    LOG_DIR = './logs'
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, "pipeline.log")
    logger = logging.getLogger("backtest_pipeline")
    logger.setLevel(logging.INFO)
    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 按天滚动的文件输出，保留最近 7000 天日志
    fh = TimedRotatingFileHandler(
        LOG_PATH, when="midnight", interval=1, backupCount=7000, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s"
    )
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    return logger, ts


MODEL_TYPE = 'dt' # dt rf xgb
TEST_FLAG = False
N_JOBS = 4
TOP_K = 15
K_FOLD = 5
ts = datetime.now().strftime("%Y%m%d_%H%M%S")


def format_seconds(seconds): # 打印时间的工具
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def day_2_trading_day(day, trading_day_list_df, logic = 'prev', cnt = 1):
    # prev:向前找还是向后找 cnt: 找第几个
    if logic != 'prev' and logic != 'forw':
        logging.info("day_2_trading_day 参数错误")
        return day
    # day_src = day
    for i in range(0, 15):  # 处理Hold集开始非交易日的情况 少了一个range!!!
        if (trading_day_list_df['date'] == day).any(): # 不能用in
            cnt -= 1
        if cnt == 0:
            break
        if logic == 'prev':
            day -= pd.Timedelta(days=1)
        else:
            day += pd.Timedelta(days=1)
    if not day in trading_day_list_df['date'] or cnt != 0:
        logging.info("day_2_trading_day 数据错误")
    # print("day2trading day", day_src ,"->>", day, "cnt= ", cnt )
    return day


def train_model_with_tscv(X_train, y_train, model_type='dt', n_splits=K_FOLD, random_seed=29, criterion = 'gini',
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
            'verbosity': 0
        }

    models = []
    scores = []
    times = []
    logger.info(f"Training {model_type.upper()} model with {n_splits}-fold TimeSeriesSplit...\n")
    for i, (train_index, val_index) in enumerate(tqdm(tscv.split(X_train), total=n_splits, desc="Progress")):
        start_time = time.time()

        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        if model_type == 'xgb':
            label_encoder = LabelEncoder()
            y_tr = label_encoder.fit_transform(y_tr)
            y_val = label_encoder.transform(y_val)

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)

        if model_type == 'xgb':
            y_pred = label_encoder.inverse_transform(y_pred)
            y_val = label_encoder.inverse_transform(y_val)

        score = accuracy_score(y_val, y_pred)

        models.append(model)
        scores.append(score)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        if logger:
            logger.info(f"Fold {i + 1}: Accuracy={score:.4f}, Time={elapsed_time:.2f} seconds")

    best_model = models[np.argmax(scores)]
    logger.info(f"\nBest accuracy: {max(scores):.4f}")
    logger.info(f"Average training time per fold: {np.mean(times):.2f} seconds")
    logger.info(f"total use time : {time.time() - start_time:.2f} seconds")
    return best_model


def evaluate_model_with_backtest(model, info_dict, df, factor_cols, return_col,
                                 top_k=TOP_K, target_label=4): # 选股回测
    test_end = info_dict["test_end"]
    hold_end = info_dict["hold_end"]
    df = df[(df['date'] >= test_end) & (df['date'] <= hold_end)]


    # 结果写入设置
    period_str = f"[test period: {info_dict['test_start']} → {info_dict['test_end']}]"

    ##########       1. 选股阶段        ###########
    # 只取test_end那一天的横截面
    test_day_data = df[df['date'] == info_dict['test_end']]
    if test_day_data.empty:
        logger.info(f"{period_str} ⚠️ test_data 中未找到日期为 {info_dict['test_end']} 的数据，跳过本轮选股。")
        return np.nan, [np.nan] * len(factor_cols)
    X_test = test_day_data[factor_cols]
    if target_label not in model.classes_:
        logger.info(f"{period_str} ⚠️ 当前模型未包含 label={target_label}，模型标签为：{model.classes_}，跳过本轮选股。")
        return np.nan, [np.nan] * len(factor_cols)
    label_index = list(model.classes_).index(target_label)
    y_pred = model.predict_proba(X_test)[:, label_index]

    ########### 存储预测准确率（新增） TODO 整合回测输出，把数值和计算全部拆到后面一个单独的part 结果展示 量化框架
    accuracy=-1
    if 'label' in test_day_data.columns:
        y_true = test_day_data['label'].values
        y_pred_labels = model.predict(X_test)
        accuracy = (y_pred_labels == y_true).mean()
        # 定义保存文件路径
        acc_file = os.path.join(fr"./result/label_accuracy_{MODEL_TYPE}_{ts}.csv")
        # 如果文件不存在，先写入表头
        if not os.path.exists(acc_file):
            with open(acc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'accuracy'])
        # 追加本次结果
        with open(acc_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([info_dict["test_start"], f"{accuracy:.4f}"])
        # —— 新增结束 ——

    pred_df = pd.DataFrame({
        'code': test_day_data['code'],
        'score': y_pred
    })
    top_stocks = pred_df.sort_values(by='score', ascending=False).head(top_k)
    selected_ids = top_stocks['code'].astype(str).tolist()
    hold_returns = df[
        (df['date'] == info_dict["hold_start"]) &
        (df['code'].astype(str).isin(selected_ids))
        ]
    # 从当天的 test_day_data 中，用 code 做索引，提取所有因子列
    features_df = test_day_data.set_index('code')  # 列名: code score 6m_return 因子列
    # 把 score 和因子值合并到一个表里
    top_with_features = top_stocks.set_index('code').join(features_df)
    # 重置索引，方便输出
    top_with_features = top_with_features.reset_index()

    ###### 加入新内容 TODO https://chatgpt.com/c/683933da-439c-8010-92c2-a1c5143b6a25
    top_with_features['hold_start'] = info_dict['hold_start']
    top_with_features['hold_end'] = info_dict['hold_end']

    # 将 df 中需要的价格信息筛选出来以提高效率
    price_data = df[['code', 'date', '股票价格', 'label']]

    # 提取 hold_start 和 hold_end 日期的价格数据
    start_price_df = price_data[price_data['date'] == info_dict['hold_start']]
    # print(start_price_df)
    end_price_df = price_data[price_data['date'] == info_dict['hold_end']]
    # print(end_price_df)
    test_end_df = price_data[price_data['date'] == info_dict['test_end']]
    # print(end_price_df)

    top_with_features['label_pred'] = model.predict(top_with_features[factor_cols])

    # 遍历 features_df 中的每一行股票代码，补全价格信息
    for idx, row in top_with_features.iterrows():
        # print("check0")
        code = row['code']
        if code in list(start_price_df['code']):
            # print("check1")
            top_with_features[top_with_features['code'] == code]['hold_start_price'] = (
                start_price_df)[start_price_df['code'] == code]['股票价格']
        if code in list(start_price_df['code']):
            # print("check2")
            top_with_features[top_with_features['code'] == code]['hold_end_price'] = (
                end_price_df)[end_price_df['code'] == code]['股票价格']
        # if code in list(test_end_df['code']):
        #     print("check3")
        #     top_with_features[top_with_features['code'] == code]['label_true'] = (
        #         test_end_df)[test_end_df['code'] == code]['label']

    # 写入文件时，把所有列都输出
    output_file = os.path.join(f"./result/top_k_stocks_{MODEL_TYPE}_{ts}.txt")

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(r'************************************************************\n')
        f.write(f"训练集: {info_dict['train_start'].date()} ➜ {info_dict['train_end'].date()} | "
        f"测试集: {info_dict['test_start'].date()} ➜ {info_dict['test_end'].date()} | "
        f"持仓期: {info_dict['hold_start'].date()} ➜ {info_dict['hold_end'].date()}\n")
        f.write(f"🔎 Top-{top_k} 选股及预测标签及因子值：\n")
        f.write(top_with_features.to_string(index=False))
        f.write('\n')

    error_prefix = f"error_{str(info_dict['test_start'])[:10].replace('-', '')}"
    if hold_returns.empty:
        logger.info(f"{period_str} ⚠️ hold_data 中未找到任何选中股票，跳过该期。")
        top_stocks.to_csv(f"./debug/{error_prefix}_no_hold_data.csv", index=False)
        avg_return = np.nan
    elif hold_returns[return_col].isnull().all():
        logger.info(f"{period_str} ⚠️ 所有选中股票在 hold_data 中 future_return 全为空，股票列表如下：")
        logger.info(top_stocks)
        hold_returns.to_csv(f"./debug/{error_prefix}_all_return_nan.csv", index=False)
        top_stocks.to_csv(f"./debug/{error_prefix}_top_stocks.csv", index=False)
        avg_return = np.nan
    else:
        valid_returns = hold_returns[return_col].dropna()
        if len(valid_returns) < 5:
            logger.info(f"{period_str} ⚠️ 有效收益样本少于5个（仅 {len(valid_returns)} 支），跳过该期。")
            hold_returns.to_csv(f"./debug/{error_prefix}_too_few_valid.csv", index=False)
            avg_return = np.nan
        else:
            avg_return = valid_returns.mean()
            logger.info(f"{period_str} ✅ 成功回测：平均收益为 {avg_return:.4f}，选股数 {len(valid_returns)}")

    return avg_return, accuracy, model.feature_importances_


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
    ##########      1. 训练期及数据确认      ##########
    train_start = row["train_date"]
    test_start = row["test_date"]
    hold_start = row["buy_date"]
    hold_end = row["end_date"]
    train_end = test_start
    test_end = hold_start

    # 转为交易日
    train_start = day_2_trading_day(train_start, df, 'forw')
    train_end = day_2_trading_day(train_end, df, 'forw')
    train_end = day_2_trading_day(train_end, df, 'prev', cnt=2)
    test_start = day_2_trading_day(test_start, df, 'forw')
    test_end = day_2_trading_day(test_end, df, 'forw')
    test_end = day_2_trading_day(test_end, df, 'prev', cnt=2)
    hold_start = day_2_trading_day(hold_start, df, 'forw')
    hold_end = day_2_trading_day(hold_end, df, 'forw')

    # 数据不足时跳过
    if hold_end > df["date"].max():
        logger.info(
            f"⚠️ 数据不足，跳过第 {ridx + 1} 轮（hold_end={hold_end.date()} 超出数据范围）"
        )
        return None

    logger.info(f"\n========== 回测第 {ridx + 1} / {total_rounds} 轮 ==========")
    logger.info(
        f"训练集: {train_start.date()} ➜ {train_end.date()} | "
        f"测试集: {test_start.date()} ➜ {test_end.date()} | "
        f"持仓期: {hold_start.date()} ➜ {hold_end.date()}"
    )

    ##########      2. 数据过滤      ##########
    # 股票池过滤 缩小数据大小
    stock_universe = set(str(row["stock_list"]).split(","))
    df = df[df[stock_id_col].isin(stock_universe)]
    logger.info(f"📊 本轮测试集股票数量：{df[stock_id_col].nunique()} 只")

    # 切片数据： 提取X_train, Y_train 切片数据
    train_data = df[(df["date"] >= train_start) & (df["date"] <= train_end)] # 潜复制
    X_train = train_data[factor_cols]
    y_train = train_data[label_col]
    logger.info("✅ 本轮训练标签种类：%s", sorted(y_train.unique()))

    ##########      3. 模型训练     ##########
    model = train_model_with_tscv(X_train, y_train, model_type=model_type)

    ##########      4. 选股与回测        ##########
    info_dict = { # 都是交易日
        'train_start': train_start,
        "train_end": train_end,
        'test_start': test_start,
        'test_end': test_end,
        'hold_start': hold_start,
        'hold_end': hold_end,
    }
    avg_return, accuracy, feat_importance = evaluate_model_with_backtest(
        model=model,
        info_dict = info_dict,
        df = df,
        factor_cols=factor_cols,
        return_col=return_col,
        top_k=TOP_K,
    )

    return {
        "result": {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "hold_start": hold_start,
            "hold_end": hold_end,
            "avg_return": avg_return,
            "accuracy": accuracy,
        },
        "importance": {
            "date": hold_start,
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
    top_k: int = TOP_K,
    model_type: str = MODEL_TYPE,
):
    """f
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
    max_round = 3 if TEST_FLAG else total_rounds

    # 缺失值填补
    # 因子列缺失值用mean填补
    inf2nan = FunctionTransformer(
        lambda X: np.where(np.isfinite(X), X, np.nan), validate=False
    )
    pre_pipe = Pipeline(
        [
            ("inf2nan", inf2nan),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    df[factor_cols] = pd.DataFrame(  # 节省内存，不再copy
        pre_pipe.fit_transform(df[factor_cols]), columns=factor_cols, index=df.index
    )

    # 并行执行每轮回测
    results_all = Parallel(n_jobs=N_JOBS)(
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
    logger.info(f"🚀 回测完成（{len(results)} 轮），总用时：{format_seconds(time_cnt)}\n")
    logger.info(f"参数：ts:{ts}, model_type:{MODEL_TYPE}, k-fold: {K_FOLD}, top-k : {TOP_K}\n")
    logger.info("***********************************\n")
    return pd.DataFrame(results), pd.DataFrame(feature_importance_list)


if __name__ == '__main__':
    os.makedirs('./debug', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    os.makedirs('./result/fig', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    args = parse_args()
    MODEL_TYPE = args.model_type
    TEST_FLAG = args.test
    N_JOBS = args.n_jobs
    TOP_K = args.top_k
    K_FOLD = args.k_fold
    logger, ts = init_logs()

    # 下载数据集
    file_id = '1pULLUf_W9KKtgrZSIYMjGVqb7BeizwZM'
    dest = './data/processed/merge_data_ret.parquet'

    if not os.path.exists(dest):
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))

        with open(dest, 'wb') as f, tqdm(
                desc="Downloading",
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


    # 1.定义列名
    factor_cols = ['momentum_6m', 'momentum_11m',
                   'marketcap',
                   'earnings-to-price', 'price-to-book', 'price-to-sales', 'operating cashflow-to-price',
                   'roe', 'earnings-per-share',
                   'investment-to-price', 'current-ratio', 'operating cashflow-to-equity', 'capex',
                   'evebit', 'evebitda', 'enterprise-value',
                   'returns_12m_lagged_12m', 'returns_12m_lagged_24m',
                   'beta_3Y_coef', 'beta_3Y']
    """
        资本支出 / 总市值
        流动比率： 流动资产（缺）/流动负债（有） T10100
        ocfp: 经营活动现金流量净额(有）/ 净资产（无？）
        capex: 资本支出/营业收入（都有）zz
        evebit: 企业价值（用哪个？）/ EBIT(季度）
        evebitda: 企业价值（用哪个？）/ EBITDA(季度）
        两个lag，两个回归
    """
    label_col = 'label'  # 分类标签：高/中/低收益（分类问题）
    return_col = 'ret_fwd_6m'  # 实际未来收益率（连续值，用于回测）
    stock_id_col = 'code'  # 股票代码 后面rename
    cols_input = factor_cols + ['date', 'code', label_col, return_col, '股票价格']

    logging.info(f"共有{len(cols_input)}个因子，因子列:{list(cols_input)}")
    logging.info(f"结果列:{return_col}")
    # 2. 读取并准备数据
    # df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), encoding='utf-8-sig', usecols=cols_input)
    df = data_loader.get_daily_price_ret_pd(usecols=cols_input)
    df['date'] = pd.to_datetime(df['date'])  # 确保日期列为 datetime 类型
    df.sort_values(['date', 'code'], inplace=True)
    # df.rename(columns={'code': 'code', 'Beta3Y_Cov_y': 'Beta3Y_Cov', 'Beta3Y_Reg_y':'Beta3Y_Reg'}, inplace=True)

    # 读取股票池数据
    # stock_list_df = pd.read_csv(os.path.join(params['data_dir'], './best_stock_window_snapshot.csv'), parse_dates=['date'])
    stock_list_df = data_loader.get_stock_list_pd()

    # 清空输出文档

    la_file = os.path.join(params['result_dir'], f"top_k_stocks_{MODEL_TYPE}_{ts}.txt")
    with open(la_file, "w", encoding="utf-8") as f:
        pass

    # 3. 执行回测流程
    backtest_df, feature_df = backtest_pipeline(
        df=df,
        factor_cols=factor_cols,
        label_col=label_col,
        return_col=return_col,
        stock_id_col=stock_id_col,
        model_type = MODEL_TYPE,
    )

    # 4. 输出回测结果与因子重要性
    backtest_df.to_csv(os.path.join(params['result_dir'], f"backtest_results_{MODEL_TYPE}_{ts}.csv"), index=False)
    feature_df.to_csv(os.path.join(params['result_dir'], f"feature_importance_time_series_{MODEL_TYPE}_{ts}.csv"),
                      index=False)

    # 5. 可视化收益曲线
    draw.draw_all(model_type = MODEL_TYPE, time_stamp = ts)
    logger.info("回测完成，结果已保存！")

