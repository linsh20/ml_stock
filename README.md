# README

## 文件结构

```
项目目录/
├── data/
│   ├── 500.csv
│   └── merge_data.csv
├── result/
├── run.py
├── prepare.py
├── zz500_merge.py
├── config.py
└── README.md
```

## 代码注释

### 数据读入

- `merge_data.csv` 日度级别的数据，后面进一步处理
- `500.csv`成分股变化数据 

### `zz500_merge.py`：处理股票列表

1. 将成分股变化数据转为成分股和时间的对应数据
2. 生成每个股票在成分股内的时间范围（备注：只考虑最早和最晚，没有考虑中间被剔除的情况）

### `prepare.py`：股票数据准备

1. `calc_ret_label`计算收益率
   - 计算未来4个月和12个月的收益率，对4个月的收益率打label
2. `calc_momentum_factor`计算动量因子
   - 计算过去6个月和11个月的动量因子
3. 计算符合条件的股票，和日期进行对应

### ``run.py``：核心流程

#### 1. 入口： `main`函数

1. 读取价格数据 (read： `merge_data_ret.csv`)，作为`df`
2. 从数据中选取作为因子的列：`factor_cols = ['资产收益率A', '息税前盈余']`
3. 使用之前根据数据+中证500筛选好的股票列表数据：`best_stock_window_snapshot.csv`，作为`stock_list_df`
4. ==调用==回测主要流程函数：`backtest_df, feature_df = backtest_pipeline()`
5. 输出结果：回测结果`backtest_df`和因子重要性`feature_df`

#### 2. 主流程：`back_test_pipeline`函数

```python
def backtest_pipeline(df, factor_cols, label_col, return_col, stock_id_col,  stock_list_df, train_years=3, test_years=1, hold_months=4, step_months=4)
```

1. 根据`stock_list_df`找到起始日期；定义回测长度
   - 数据长度划分说明：`train`为3年（252d\*3），进行K-Fold模型训练；`test`为1年（252d），进行选股；`hold`为4个月（21d\*4），计算收益
2. 在数据对应的日期范围内进行滚动回测，步长为4个月：
   - 数据说明：对于`train`，使用时间范围内的全部股票（不能保证每支股票均在时间范围内都有数据，对于数据缺失问题存疑）；对于`test`和`hold`，仅使用`stock_list_df`中筛选出的股票数据
   - 缺失值处理：基于`train`生成`impute`，用于为`train`、`test`和`hold`填补缺失值。缺失值填补使用训练集中所有股票、所有日期的平均值（mean）。**只处理因子列。**
   - 训练模型：对于`train`，==调用==训练函数`model = train_model_with_tscv()`
   - 选股和回测：==调用==回测函数`avg_return, feat_importance = select_stocks_and_backtest()`

#### 3. 训练： `train_model_with_tscv`函数

```python
def train_model_with_tscv(X_train, y_train, factor_cols, model_type='rf', n_splits=10)
```

1. 选择模型类型
2. 使用TimeSeries-KFold训练
3. 返回模型

#### 4. 测试和回测：`select_stocks_and_backtest`函数

```python
def select_stocks_and_backtest(model, X_test, hold_data, return_col, imputer, stock_ids, top_k=15, test_start=None, hold_end=None)
```

1. 使用训练好的模型，在`test`集上预测出收益为最高类别的概率，记为`score`，每只股票、每天生成一个预测分数，对每只股票在所有日期上的分数进行平均，按平均后的分数从高到低选择`topK=15`支股票
2. 查找这些股票在`hold`集上的收益率，取平均（因为假设平均持仓）；检查有效股票数量
3. 返回收益率平均、模型的特征重要性

## 结果展示

1. 满足条件的股票数量

   ![image-20250506211754414](C:\Users\linsh\AppData\Roaming\Typora\typora-user-images\image-20250506211754414.png)

2. 只使用两个动量因子的回测结果 ['11m_mom', '6m_mom']
   ![image-20250506212258527](C:\Users\linsh\AppData\Roaming\Typora\typora-user-images\image-20250506212258527.png)

## 运行流程

1. 安装对应的包
2. 运行`zz500_merge.py`
3. 运行`prepare.py` 
4. 运行`run.py`