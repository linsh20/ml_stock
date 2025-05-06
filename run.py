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


def train_model_with_tscv(X_train, y_train, factor_cols, model_type='rf', n_splits=10):
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

    # TimeSeries K-Fold
    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # X_tr,  imputer = clean_nan(X_tr,  strategy='mean')
        # X_val = pd.DataFrame(imputer.transform(X_val[factor_cols]), columns=factor_cols, index=X_val.index)

        model = model_class()
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        models.append(model)
        scores.append(score)

    # 返回得分最高的模型
    best_model = models[np.argmax(scores)]
    return best_model


def select_stocks_and_backtest(model, test_data, hold_data, factor_cols, return_col,
                                imputer, top_k=15, test_start=None, test_end=None, hold_end=None, target_label=5):
    period_str = f"[test period: {test_start} → {test_end}]"

    # 选股阶段
    X_test = test_data[factor_cols]
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 模型是否包含目标标签？
    if target_label not in model.classes_:
        print(f"{period_str} ⚠️ 当前模型未包含 label={target_label}，模型标签为：{model.classes_}，跳过本轮选股。")
        return np.nan, [np.nan] * len(factor_cols)

    # 查找目标 label 的列索引
    label_index = list(model.classes_).index(target_label)
    y_pred = model.predict_proba(X_test)[:, label_index]

    # 1. 构建原始预测 DataFrame
    pred_df = pd.DataFrame({
        'stock_id': test_data['stock_id'],
        'score': y_pred
    })

    # 2. 对每只股票聚合（例如取平均预测得分）
    agg_pred_df = pred_df.groupby('stock_id', as_index=False)['score'].mean()

    # 3. 按聚合后的得分排序，选出 top_k
    top_stocks = agg_pred_df.sort_values(by='score', ascending=False).head(top_k)
    selected_ids = top_stocks['stock_id'].astype(str).tolist()

    # 打印 top_k 股票及其预测标签
    print(f"🔎 Top-{top_k} 选股及预测标签：")
    print(top_stocks[['stock_id', 'score']])

    # 查未来收益
    hold_returns = hold_data[hold_data['stock_id'].astype(str).isin(selected_ids)]

    # 保存异常情况
    error_prefix = f"error_{str(test_start)[:10].replace('-', '')}"
    if hold_returns.empty:
        print(f"{period_str} ⚠️ hold_data 中未找到任何选中股票，跳过该期。")

        # 保存 top_k 股票及其得分
        top_stocks.to_csv(f"./debug/{error_prefix}_no_hold_data.csv", index=False)

        avg_return = np.nan

    elif hold_returns[return_col].isnull().all():
        print(f"{period_str} ⚠️ 所有选中股票在 hold_data 中 future_return 全为空，股票列表如下：")
        print(top_stocks)

        # 保存 hold_returns 的原始内容
        hold_returns.to_csv(f"./debug/{error_prefix}_all_return_nan.csv", index=False)
        top_stocks.to_csv(f"./debug/{error_prefix}_top_stocks.csv", index=False)

        avg_return = np.nan

    else:
        valid_returns = hold_returns[return_col].dropna()
        if len(valid_returns) < 5:
            print(f"{period_str} ⚠️ 有效收益样本少于5个（仅 {len(valid_returns)} 支），跳过该期。")

            # 保存
            hold_returns.to_csv(f"./debug/{error_prefix}_too_few_valid.csv", index=False)
            avg_return = np.nan
        else:
            avg_return = valid_returns.mean()
            print(f"{period_str} ✅ 成功回测：平均收益为 {avg_return:.4f}，选股数 {len(valid_returns)}")

    return avg_return, model.feature_importances_


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

    while start_idx + train_days + test_days + hold_days <= len(dates):
        # 计算时间范围
        train_start = dates[start_idx] # 3年
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days] # 1年
        test_end = dates[start_idx + train_days + test_days - 1]
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
        # 缺失值处理->生成impute
        X_all_for_impute = X_train.copy()
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
        model = train_model_with_tscv(X_train, y_train, factor_cols, model_type='dt')

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
            hold_end=hold_end
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

    return pd.DataFrame(results), pd.DataFrame(feature_importance_list)


def plot_cumulative_return(backtest_df):
    backtest_df['cum_return'] = (1 + backtest_df['avg_return']).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(backtest_df['test_period_start'], backtest_df['cum_return'], label='Cumulative Return', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Performance')
    plt.grid(True)

    # 添加数据点标签
    for i in range(len(backtest_df)):
        x = backtest_df['test_period_start'].iloc[i]
        y = backtest_df['cum_return'].iloc[i]
        plt.scatter(x, y, color='red')  # 标出数据点
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
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


def clean_nan(X, strategy='mean'):  # 废弃
    if strategy not in ['mean', 'median']:
        raise ValueError("strategy must be 'mean' or 'median'")

    imputer = SimpleImputer(strategy=strategy)
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X_clean, imputer

def test_get_stock_list_for_date():  # 测试函数
    # 读取股票池快照数据
    stock_list_df = pd.read_csv(
        os.path.join(params['data_dir'], './best_stock_window_snapshot.csv'),
        parse_dates=['date']
    )

    # 指定要测试的日期
    test_date = pd.to_datetime('2020-05-01')

    # 调用主函数
    stock_set = get_stock_list_for_date(test_date, stock_list_df)

    # 打印结果
    print(f"股票池日期 <= {test_date} 最近的一期包含 {len(stock_set)} 支股票。")
    print("前10只股票代码如下：")
    print(sorted(list(stock_set))[:10])  # 显示前10只股票代码



if __name__ == '__main__':
    # test_get_stock_list_for_date()
    # 1. 读取并准备数据
    df = pd.read_csv(os.path.join(params['data_dir'], 'merge_data_ret.csv'), encoding='utf-8-sig')  # 假设已有整理好的特征与标签数据
    df['日期'] = pd.to_datetime(df['日期'])  # 确保日期列为 datetime 类型
    df.sort_values(['日期', '证券代码'], inplace=True)

    df.rename(columns={'日期': 'date', '证券代码': 'stock_id'}, inplace=True)

    # 2. 定义列名
    factor_cols = ['6m_return', '11m_return']
    # factor_cols = [
    #     '股票总市值', '股票价格', '每股收益', '每股净资产',
    #     '每股主营业务收入', '每股经营现金流', '现金流比股价',
    #     '净资产增长率', '主营业务收入增长率', '现金流增长率',
    #     '企业价值含货币资金', '企业价值不含货币资金', '企业倍数',
    #     'pe', 'pb', 'pcf', 'ps',
    #     'circulatedmarketvalue', 'liquidility', 'turnover'
    # ]

    label_col = 'label'  # 分类标签：高/中/低收益（分类问题）
    return_col = 'ret_fwd_4m'  # 实际未来收益率（连续值，用于回测）
    stock_id_col = 'stock_id'  # 股票代码

    # 读取股票池数据
    stock_list_df = pd.read_csv(os.path.join(params['data_dir'], './best_stock_window_snapshot.csv'), parse_dates=['date'])

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
        step_months=4,
    )

    # 4. 输出回测结果与因子重要性
    backtest_df.to_csv(os.path.join(params['result_dir'],'backtest_results.csv'), index=False)
    feature_df.to_csv(os.path.join(params['result_dir'],'feature_importance_time_series.csv'), index=False)

    # 5. 可视化收益曲线
    plot_cumulative_return(backtest_df)

    print("回测完成，结果已保存！")
