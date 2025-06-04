import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('data/BND_TreasYield.csv')

# 筛选出 Cvtype=2 且 Yeartomatu=0.5 的行
filtered_df = df[(df['Cvtype'] == 2) & (df['Yeartomatu'] == 0.5)]

# 输出到新的CSV文件
filtered_df.to_csv('data/BND_TreasYield_filter.csv', index=False)

print("筛选完成，文件已保存为 data/BND_TreasYield_filter.csv")
