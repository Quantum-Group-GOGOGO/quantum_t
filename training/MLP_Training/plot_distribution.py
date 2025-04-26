import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import pandas as pd


data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
prediction_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions_120to80_2LSTM_future1.pkl"

df = pd.read_pickle(prediction_path)
split_index = int(len(df) * 0.9)
#split_index2 = int(len(df) * 0.9)
df = df.iloc[split_index:]


x = df['evaluation_120'].values
y = df['prediction3'].values
corr = np.corrcoef(x, y)[0,1]
print("Pearson r =", corr)
plt.figure(figsize=(6,5))
hb = plt.hexbin(
    x, y,
    gridsize=40,                  # 网格分辨率
    #extent=(-5, 5, -5, 5),         # x,y 都限制在 [-5,5]
    mincnt=1,                      # 忽略空格子
    cmap='coolwarm'                 # 配色，可选
)
plt.colorbar(hb, label='点数')
plt.xlabel('evaluation_120')
plt.ylabel('prediction3')
plt.title('Hexbin Joint Distribution')
plt.clim(0, 500)
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.tight_layout()
plt.show()