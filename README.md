## 文件目录

📦 code/
├── 📁 data/
│   ├── 📁 financial/  # 财务报表
│   └── 📁 price/ # 日度级别价格数据
│       ├── 📁 back_adj/ # 后复权
│       ├── 📁 forw_adj/ # 前复权
│       ├── 📁 normal/ # 无复权
│       └── 📄 csi500_stock_list.csv
├── 📁 factors/  # 存储计算出的因子数据
├── 📄 config.py
├── 📄 prepare.py
└── 📄 factor_gen.py

## 运行流程

1. 使用prepare.py 下载所需的数据
2. 使用factor_gen.py生成因子