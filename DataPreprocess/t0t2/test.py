import pandas as pd

# 创建样例 DataFrame
data = {'close': [12, 9, 15, 8, 20, 7]}
df = pd.DataFrame(data)

# 使用 for 循环遍历 'close' 列
for i in range(len(df)):
    if df['close'].iloc[i] < 10:
        print("hello")