import pandas as pd
import numpy as np
from tqdm import tqdm
from ydata_profiling import ProfileReport

def exp_norm(df, column_name, scaling_factor=1e5):
    """
    对指定列进行指数归一化操作，直接覆盖原列.
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame
    column_name (str): 需要进行归一化的列名
    scaling_factor (float): 缩放因子, 默认为 1e5

    返回:
    pd.DataFrame: 更新后的 DataFrame, 指定列被覆盖
    """
    df[column_name] = np.exp(-df[column_name] / scaling_factor)
    return df

def logi_norm(df, column_name, scaling_factor=1):
    """
    使用带除法缩放因子的 Logistic 函数将指定列归一化到 [0, 1] 范围.
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame
    column_name (str): 需要归一化的列名
    scaling_factor (float): 缩放因子, 用于控制函数的陡峭程度

    返回:
    pd.DataFrame: 更新后的 DataFrame, 指定列被覆盖
    """
    df[column_name] = 1 / (1 + np.exp(-df[column_name] / scaling_factor))
    return df

def logi_rescale(df, column_name, scaling_factor=1):
    """
    通过 Logit 函数将数据从 [0, 1] 投影到 [-inf, +inf]，再通过调整缩放因子重新投影回 [0, 1]。
    直接覆写原来的列，不创建临时列.

    scaling_factor<1会让数据更加集中在0.5附近,scaling_factor>1则会让数据更加集中在0或者1附近
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame
    column_name (str): 要重新投影的列名
    scaling_factor (float): 缩放因子, 用于控制数据分布形态, 默认为 1

    返回:
    pd.DataFrame: 更新后的 DataFrame
    """
    # 内部的 Sigmoid 函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 内部的 Logit 函数
    def logit(x):
            return np.log(x / (1 - x))

    # 1. 通过 Logit 函数投影到 [-inf, +inf]，直接覆写原来的列
    df.loc[df[column_name] > 1-1e-9, column_name] = 1-1e-9
    df.loc[df[column_name] < 1e-9, column_name] = 1e-9
    df[column_name] = logit(df[column_name])
    df[column_name] = df[column_name] / scaling_factor
    # 2. 通过 Sigmoid 函数重新投影回 [0, 1]，并调整缩放因子
    df[column_name] = sigmoid(df[column_name])

    return df

#data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
data_base='D:\quantum\quantum_t_data\quantum_t_data'

T5_data_path=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated.pkl'
T5_data_path_test=data_base+'/type5/Nasdaq_qqq_align_labeled_base_evaluated_test.pkl'

T6_data_path=data_base+'/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'

df=pd.read_pickle(T5_data_path)
print(df['volume'].head())
print(df.columns)
#print(df.describe())

#指数归一化
#sf越大，归一化越靠近1
#sf越小，归一化越靠近0
exp_norm(df, 'volume', scaling_factor=1e4)
exp_norm(df, 'volume_10', scaling_factor=1e4)
exp_norm(df, 'volume_60', scaling_factor=1e4)
exp_norm(df, 'volume_240', scaling_factor=1e4)
exp_norm(df, 'volume_1380', scaling_factor=5e4)

exp_norm(df, 'pre_event', scaling_factor=1e5)
exp_norm(df, 'post_event', scaling_factor=1e5)
exp_norm(df, 'pre_break', scaling_factor=4e3)
exp_norm(df, 'post_break', scaling_factor=4e3)
exp_norm(df, 'absolute_time', scaling_factor=8e6)

#logistic函数归一化
#sf越大，归一化越靠中间
#sf越小，归一化越分散
#logi_norm(df, 'evaluation_30', scaling_factor=1e0)
#logi_norm(df, 'evaluation_60', scaling_factor=2e0)
#logi_norm(df, 'evaluation_120', scaling_factor=4e0)
#logi_norm(df, 'evaluation_300', scaling_factor=4e0)
#logi_norm(df, 'evaluation_480', scaling_factor=1e1)

#logistic函数归一化
#sf越大，归一化越靠中间
#sf越小，归一化越分散
#logi_norm(df, 'evaluation_30h', scaling_factor=1e0)
#logi_norm(df, 'evaluation_60h', scaling_factor=2e0)
#logi_norm(df, 'evaluation_120h', scaling_factor=4e0)
#logi_norm(df, 'evaluation_300h', scaling_factor=4e0)
#logi_norm(df, 'evaluation_480h', scaling_factor=1e1)

#重新调整数据分布
#sf<1e0 越小越扩散到两头
#sf>1e0 越大越集中到中间
logi_rescale(df, 'volume', scaling_factor=2e0)
logi_rescale(df, 'volume_10', scaling_factor=5e0)
logi_rescale(df, 'volume_60', scaling_factor=6e0)
logi_rescale(df, 'volume_240', scaling_factor=4e0)
logi_rescale(df, 'volume_1380', scaling_factor=1e0)

#视情况是否需要摒弃volume_YEAR
df.drop('volume_YEAR', axis=1, inplace=True)
#丢弃其它不需要的辅助数据
df.drop('time_break_flag', axis=1, inplace=True)
df.drop('event', axis=1, inplace=True)
df.drop('time_fraction', axis=1, inplace=True)
df.drop('week_fraction', axis=1, inplace=True)

pd.options.display.float_format = '{:.2e}'.format

print(df['volume'].head())
df.to_pickle(T6_data_path)