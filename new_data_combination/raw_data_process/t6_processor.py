from env import *
from preallocdataframe import PreallocDataFrame
from tqdm import tqdm
import pandas as pd
import numpy as np
from neuro_network_processor import live_nn
def exp_norm1(x, scaling_factor=1e5):
    return np.exp(-x / scaling_factor)

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

def logi_norm1(x, scaling_factor=1):
    return  1 / (1 + np.exp(-x / scaling_factor))

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

def logi_rescale1(x, scaling_factor=1):
    # 内部的 Sigmoid 函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 内部的 Logit 函数
    def logit(x):
            return np.log(x / (1 - x))
    
    if x>1-1e-9:
        x=1-1e-9
    if x<1e-9:
        x=1e-9
    y=logit(x)
    return sigmoid(y/scaling_factor)

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

class live_t6:
    def __init__(self):
        self.t3FilePath=live_data_base+'/type3/type3Base.pkl'
        self.t6FilePath=live_data_base+'/type6/type6Base.pkl'
        self.initial_dataBase()

    def link_t6t3sub(self, t3_processor):
        self.t3_p=t3_processor
        self.t3_p.link_t6obj(self)
        self.t3Base=self.t3_p.t3Base
        self.T6Renew()

    def link_nnobj(self, nn_processor:live_nn):#只允许被link_sub函数调用
        self.nn_p=nn_processor

    def initial_dataBase(self):
        self.t3Base= PreallocDataFrame(pd.read_pickle(self.t3FilePath))
        if os.path.exists(self.t6FilePath):
            print('更新T6数据')
            self.t6Base = PreallocDataFrame(pd.read_pickle(self.t6FilePath))
            self.T6Renew()
            self.t6Base.to_pickle(self.t6FilePath)

        else:
            print('T6数据缺失，正在整理')
            self.t6Base=self.processT3Base(self.t3Base)
            self.t6Base.to_pickle(self.t6FilePath)

    def liveInitial(self):
        self.initial_dataBase()

    async def liveRenew(self,decide):
        self.T6Renew()
        await self.nn_p.liveRenew(decide)
        
    def T6Renew(self):
        # 取出 t6Base 最后一行的时间
        last_t6_ts = self.t6Base.index[-1]

        # 如果要包含等于 last_t6_ts 的那一行：
        processing_df = self.t3Base.loc[last_t6_ts + pd.Timedelta('1ns'):].copy()
        
        if len(processing_df) == 0:
            if self.t6Base.index[-1] == self.t3Base.index[-1]:
                print('T6数据完全，无需更新')
                return
            else:
                print('警告，T3更新后T6无法更新，最晚时间早于T3')
                print(self.t3Base.tail())
                print(self.t6Base.tail())
                return
        
        print(f'T6 {len(processing_df)} 行数据更新')
        for i, (idx, row) in enumerate(processing_df.iterrows()):
            volumeO=row['volume']
            volume= exp_norm1(row['volume'], scaling_factor=1e4)
            volume_10= exp_norm1(row['volume_10'], scaling_factor=1e4)
            volume_60= exp_norm1(row['volume_60'], scaling_factor=1e4)
            volume_240= exp_norm1(row['volume_240'], scaling_factor=1e4)
            volume_1380= exp_norm1(row['volume_1380'], scaling_factor=5e4)


            pre_event= exp_norm1(row['pre_event'], scaling_factor=1e5)
            post_event= exp_norm1(row['post_event'], scaling_factor=1e5)
            pre_break= exp_norm1(row['pre_break'], scaling_factor=4e3)
            post_break= exp_norm1(row['post_break'], scaling_factor=4e3)
            absolute_time= exp_norm1(row['absolute_time'], scaling_factor=8e6)


            volume=logi_rescale1(volume, scaling_factor=2e0)
            volume_10=logi_rescale1(volume_10, scaling_factor=5e0)
            volume_60=logi_rescale1(volume_60, scaling_factor=6e0)
            volume_240=logi_rescale1(volume_240, scaling_factor=4e0)
            volume_1380=logi_rescale1(volume_1380, scaling_factor=1e0)

            self.t6Base.append_row_keep_last(idx, [row['open'], row['high'], row['low'], row['close'], row['volume'],pre_event,post_event,pre_break,post_break,
                                                   row['sinT'],row['cosT'],row['week_fraction_sin'],row['week_fraction_cos'],absolute_time,
                                                   volume_10,volume_60,volume_240,volume_1380,
                                                   row['close_10'],row['close_60'],row['close_240'],row['close_1380'],
                                                   row['log_ret'],row['volweek_raw'],row['close3d'],
                                                   volumeO])

    def processT3Base(self,processing_t3in):
        processing_t3=processing_t3in.copy()
        processing_t3['volumeO']=processing_t3['volume'].copy()
        exp_norm(processing_t3, 'volume', scaling_factor=1e4)
        exp_norm(processing_t3, 'volume_10', scaling_factor=1e4)
        exp_norm(processing_t3, 'volume_60', scaling_factor=1e4)
        exp_norm(processing_t3, 'volume_240', scaling_factor=1e4)
        exp_norm(processing_t3, 'volume_1380', scaling_factor=5e4)

        exp_norm(processing_t3, 'pre_event', scaling_factor=1e5)
        exp_norm(processing_t3, 'post_event', scaling_factor=1e5)
        exp_norm(processing_t3, 'pre_break', scaling_factor=4e3)
        exp_norm(processing_t3, 'post_break', scaling_factor=4e3)
        exp_norm(processing_t3, 'absolute_time', scaling_factor=8e6)

        logi_rescale(processing_t3, 'volume', scaling_factor=2e0)
        logi_rescale(processing_t3, 'volume_10', scaling_factor=5e0)
        logi_rescale(processing_t3, 'volume_60', scaling_factor=6e0)
        logi_rescale(processing_t3, 'volume_240', scaling_factor=4e0)
        logi_rescale(processing_t3, 'volume_1380', scaling_factor=1e0)

        return PreallocDataFrame(processing_t3)