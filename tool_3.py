import pandas as pd
import shutil

# 文件路径
original_file = 'data/processed/merge_data_ret.parquet'
backup_file = 'data/processed/merge_data_ret_before_drop.parquet'

# # 备份原文件
# shutil.copyfile(original_file, backup_file)
#
# # 读取原始文件
# df = pd.read_parquet(original_file)
#
# # 删除'label'列中值为1的行
# df_filtered = df[df['label'] != 1]
#
# # 保存修改后的DataFrame到原始路径
# df_filtered.to_parquet(original_file, index=False)

# 备份原文件
try:
    shutil.copyfile(original_file, backup_file)
    print(f"✅ 已成功备份文件到: {backup_file}")
except Exception as e:
    print(f"❌ 备份文件失败: {e}")
    exit()

# 读取原始数据
try:
    df = pd.read_parquet(original_file)
    print(f"📊 原始数据行数: {len(df)}")
except Exception as e:
    print(f"❌ 读取原始文件失败: {e}")
    exit()

# 删除 label == 1 的行
df_filtered = df[df['label'] != 1]
removed_rows = len(df) - len(df_filtered)
print(f"🧹 删除了 {removed_rows} 行 (label == 1)")

# 保存新数据
try:
    df_filtered.to_parquet(original_file, index=False)
    print(f"✅ 处理后数据已保存到原文件: {original_file}")
except Exception as e:
    print(f"❌ 保存处理后文件失败: {e}")
    exit()

# 最终验证
try:
    df_check = pd.read_parquet(original_file)
    if (df_check['label'] == 1).any():
        print("⚠️ 警告：处理后文件中仍包含 label == 1 的行")
    else:
        print("🎉 检查通过：label == 1 的行已全部删除")
except Exception as e:
    print(f"❌ 验证处理结果时出错: {e}")