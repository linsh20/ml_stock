# 代码说明和结果展示

## 文件结构

```
文件目录:.
│  config.py
│  data.py
│  draw.py
│  get_fin_rep.py
│  prepare.py
│  README.md
│  read_csv.py
│  run.py
│  因子解释.csv
├─data
│  │  905_daily_fill.csv
│  │  905_price.csv
│  ├─financial
│  │  │  000005_balance.csv
│  │  │  ......
│  ├─processed
│  │      merge_data_ret.parquet
│  │      zz500_list.csv
│  │      zz500_list_filter.csv
│  ├─raw
│  │      merge_final.dta
│  │      merge_final.parquet
│  │      season_500_0512.csv
│  │      season_500_0512.parquet
│  └─read_csv
├─debug
├─result
│      backtest_results_dt.csv
│      feature_importance_time_series_dt.csv
│      label_accuracy.csv
│      label_accuracy_dt.csv
│      top_k_stocks_dt.txt
├─src
│  │  data_loader.py
│  │  preprocess.py
```

## step 1 - 数据清洗

### 数据读入

- `.data/raw/merge_final.dta` ：股票日度级别的数据，来源：国泰君安数据库；
- `./data/raw/season_500_0512.csv`：股票财报数据等季度级别的数据，来源：国泰君安数据库；
- `./data/500.csv`：成分股变化数据
- `./data/financial/*_balance.csv`：财报数据，主要为了获取财报发布日期。来源：akshare新浪财报api。

### 步骤

1. 配置环境，安装对应的包
2. \*运行`data_loader.py`，把数据转为更高效的parquet格式，方便后续处理
3. 运行`preprocess.py`的两个函数，数据对齐
4. 运行`prepare.py`计算因子（耗时较长）
5. 得到`./data/processed/merge_data_ret.parquet`，作为之后模型训练的输入

## step 2 - 模型训练

### 数据读入

- `./data/processed/merge_data_ret.parquet`，合并了日度、季度，考虑了财报发布时间的准point-in-time数据，也计算了因子。
- `./data/processed/zz500_list_filter.csv`，对齐中证500成分股调整日的滑动窗口股票列表
- `./data/FBND_TreasYield_filter.csv`，半年期无风险利率

### 运行流程

6. 运行`main_model.py`，训练完成后也会自动调用`draw.py`画图（也可以自行单独调用）
```angular2html
参数说明：
model_type：可以填入dt rf xgb。默认为dt.
n_jobs:cpu线程并行数量，调大能缩短训练时间，但是如果过大会爆内存。默认为8。
top_k:持仓股票数。默认为15.
k_fold:训练-验证集划分的折数。默认为5.

如果用ide(如pycharm）直接运行，就是默认参数。可以在代码的一开始直接改default的值.
也可以用命令行运行直接设置参数，命令行运行示例：
python main_model.py --model_type dt --n_jobs 8 --top_k 15 --k_fold 5
```
## 代码注释

### `src/preprocess.py`：数据预处理流程

数据预处理函数：

- `daily_data_2_stock_list()`函数：从中证500每日成分股数据`905_daily_fill.csv`，生成阶段数据，并进行筛选，生成`zz500_list.csv`和`zz500_list_filter.csv`（滑动窗口结果）

- `preprocess_daily_season_data()`函数：按point-in-time逻辑合并日度数据和季度数据；从akshare下载的财报中获取财报公告日期，将公告日期的下一天作为信息可用日

### `prepare.py`：股票数据清洗

输入：日度和季度原始数据、每支股票在中证500内的时间范围；输出：因子、符合交易期间长度的日期和股票列表

2. 计算因子
   1. `calc_ret_label`：计算收益率
      - 计算未来4个月和12个月的收益率，对4个月的收益率打label
   2. `calc_momentum_factor`：计算动量因子和Lagged_return
      - 计算过去6个月和11个月的动量因子
      - Lagged_return为过去X日的12个月收益率
   3. `create_factors`：处理其他因子，主要涉及对excel列之间的运算
   4. `calc_beta_3y_factors`：从中证500指数引入市场收益率，计算`beta_cov`和`beta_reg`

备注：缺失值的处理在`main_model.py`中进行

废弃部分：

- `merge_season_data`：将季度数据所需列和日度数据进行对应，整合进日度数据表中
- 计算符合条件的股票，和日期进行对应
  1. `calc_period`：结合中证500数据和价格数据，生成每支股票可用日期范围 `filtered_stock_date_range.csv`
  2. `period2cnt`：计算每日数据可用的股票数量，画图可视化
  3. `get_date_list`：在`calc_period`结果的基础上，遍历4个月（84d），计算最佳开始日（最大化股票总数量），输出`best_stock_window_snapshot.csv`，为开始日和股票List的匹配结果

### ``main_model.py``：模型训练和回测

输入：清洗好的数据；输出：结果

#### 1. 入口： `main`函数

1. 读取价格数据 (read： `merge_data_ret.csv`)，作为`df`
2. 从数据中选取作为因子的列：`factor_cols = ['6m_return', '11m_return']`
3. 使用之前根据数据+中证500筛选好的满足条件的股票列表数据：`zz500_list_filter.csv`，作为`stock_list_df`
4. ==调用==回测主要流程函数：`backtest_pipeline()`
5. 输出结果：回测结果`backtest_df`和因子重要性`feature_df`

#### 2. 主流程：`back_test_pipeline`函数

（使用多线程加快训练速度，对内部进行进一步封装）

```python
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
def train_model_with_tscv(X_train, y_train, model_type='rf', n_splits=10)
```

1. 选择模型类型
2. 使用TimeSeries-KFold训练
3. 返回模型

#### 4. 测试和回测：`evaluate_model_with_backtest`函数

```python
def select_stocks_and_backtest(model, X_test, hold_data, return_col, imputer, stock_ids, top_k=15, test_start=None, hold_end=None)
```

1. 使用训练好的模型，在`test`集上预测出收益为最高类别的概率，记为`score`，每只股票、每天生成一个预测分数，对每只股票在所有日期上的分数进行平均，按平均后的分数从高到低选择`topK=15`支股票
2. 查找这些股票在`hold`集上的收益率，取平均（因为假设平均持仓）；检查有效股票数量
3. 返回收益率平均、模型的特征重要性

### `draw.py`画图和计算指标
每期收益图、累计收益图、标签预测准确性图、因子重要性图、指标

### 其他文件

- `src/data_loader.py`：统一数据读入接口（函数调用）；数据格式转换工具（main函数)
- `read_csv.py`：用来根据列字段（如股票代码或日期）筛选csv/parquet中的行，导出可以查看的新文件，解决csv/parquet格式大文件因过大而不便查看的问题
- `get_fin_rep.py`：使用akshare api下载资产负债表
- `config.py`：用来保存参数，比如文件路径

### 废弃的函数（存档）

#### `zz500_merge.py`：处理股票列表

输入：中证500成分股变化数据；输出：每只股票在中证500内的时间范围

1. 将成分股变化数据转为成分股和时间的对应数据
2. 生成每个股票在成分股内的时间范围（备注：要求所有交易日均在中证500组合内）

废弃原因：原本是4个月步长，现在直接改为和成分股调整日对齐，逻辑换了

## 结果影响因素

top-K：1

k-fold： 试了5和10，结果差别很小

model-type： xgb > rf > dt

random seed ： 会有影响，当前固定的29（可以修改`random_seed`变量调整），可以复现

## 可视化展示

### 数据部分

1. 满足条件的股票数量（数据和中证500取交集，要求所有时间都在中证500内）

### 训练部分

训练时间

### 结果部分

1. 累计收益
   
呈现：累计收益（折线图）；与中证500指数对比（折线图）
   
   数据：股票购买日 - 股票组合`fwd_6m_return`  - 中证500 `fwd_6m_return`【？】
   
2. 每个阶段的收益

   呈现：（条形图）与中证500对比（折线图）

3. feature importance（箱图或折线图）

4. label accuracy

## 代码统计
### 代码行数
python 2000-3000行 (pycharm statistic工具统计)

### 版本记录
见git
