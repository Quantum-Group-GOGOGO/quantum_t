import pandas as pd
import numpy as np
from tqdm import tqdm
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
T6_data_path=data_base+'/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'

df=pd.read_pickle(T6_data_path)

# 绘制单列的分布图，比如 'volume' 列
#sns.histplot(df['volume'], kde=True)
#sns.histplot(df['volume_10'], kde=True)
#sns.histplot(df['volume_60'], kde=True)
#sns.histplot(df['volume_240'], kde=True)
#sns.histplot(df['volume_1380'], kde=True)

#sns.histplot(df['pre_event'], kde=True)
#sns.histplot(df['post_event'], kde=True)
#sns.histplot(df['pre_break'], kde=True)
#sns.histplot(df['post_break'], kde=True)
#sns.histplot(df['absolute_time'], kde=True)

#sns.histplot(df['evaluation_30'], kde=True)
#sns.histplot(df['evaluation_60'], kde=True)
#sns.histplot(df['evaluation_120'], kde=True)
#sns.histplot(df['evaluation_300'], kde=True)
#sns.histplot(df['evaluation_480'], kde=True)
#plt.show()

# 生成报告
profile = ProfileReport(df)

# 保存为 HTML 文件
report_file = "report.html"
profile.to_file(report_file)