import pandas as pd

# 假设你已经加载了 DataFrame 'df'
data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
#data_base='D:\quantum\quantum_t_data\quantum_t_data'
T6_data_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'

df = pd.read_pickle(T6_data_path)

# 确定分割位置
split_index = int(len(df) * 0.9)  # 分割点在整个数据集的 90%

# 分割数据集
train_df = df.iloc[:split_index]  # 前 90% 的数据作为训练集
test_df = df.iloc[split_index:]   # 后 10% 的数据作为测试集

# 保存训练集和测试集
train_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_train.pkl'
test_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_test.pkl'

train_df.to_pickle(train_path)
test_df.to_pickle(test_path)

print(f"训练集已保存到: {train_path}")
print(f"测试集已保存到: {test_path}")