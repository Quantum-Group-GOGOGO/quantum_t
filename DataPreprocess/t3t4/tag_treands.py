import printh as ph
import pandas as pd
import numpy as np
from tqdm import tqdm
def addtags(df,increase_value,decrease_value,evaluation_length):
    """
    给定 original df（包含 high, low, close 列）、
    上涨阈值 increase_value（如 0.02 表示 2%）、
    下跌阈值 decrease_value（如 0.01 表示 1%）、
    向后观测长度 evaluation_length，
    生成新列 'tag'：
      1  ：先达到涨幅阈值
     -1  ：先达到跌幅阈值
      0  ：evaluation_length 内两端都没到
    同一行同时满足涨跌阈值时，按涨幅（1）标记。
    """
    n = len(df)
    highs  = df['high'].to_numpy()
    lows   = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    
    tags = np.zeros(n, dtype=int)
    tags_in = np.zeros(n, dtype=int)
    tags_flat = np.zeros(n, dtype=int)
    tags_de = np.zeros(n, dtype=int)
    increase_num=0
    decrease_num=0
    for i in tqdm(range(n-evaluation_length), desc="计算标签"):
        A = closes[i]
        up_thr   = A * (1 + increase_value)
        down_thr = A * (1 - decrease_value)
        
        tag = 1
        tag_in = 0
        tag_flat = 1
        tag_de = 0
        # 向后看 evaluation_length 根 bar
        for j in range(1, evaluation_length + 1):
            if i + j >= n:
                break
            h = highs[i + j]
            l = lows[i + j]
            # 同行同时满足，则按 1
            if h > up_thr and l < down_thr:
                tag = 2
                tag_in = 1
                tag_flat = 0
                increase_num += 1
                break
            # 先到达哪一端就标哪一端
            if h > up_thr:
                tag = 2
                tag_in = 1
                tag_flat = 0
                increase_num += 1
                break
            if l < down_thr:
                tag = 0
                tag_de = 1
                tag_flat = 0
                decrease_num += 1
                break
        
        tags[i] = tag
        tags_in[i] = tag_in
        tags_flat[i] = tag_flat
        tags_de[i] = tag_de
    
    df = df.copy()
    df['tag'] = tags
    df['tags_in'] = tags_in
    df['tags_flat'] = tags_flat
    df['tags_de'] = tags_de
    print(increase_num)
    print(decrease_num)
    return df
    

if __name__ == "__main__":
    #data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
    T4_data_path = data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated_history.pkl'
    T4_data_tag_path =  data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated_history_tag.pkl'
    df = pd.read_pickle(T4_data_tag_path)
    increase_value =0.1/100
    decrease_value =0.1/100
    evaluation_length =20
    #print(df['tag'].head(100510))
    df_tag = addtags(df,increase_value,decrease_value,evaluation_length)
    df_tag.to_pickle(T4_data_tag_path)
    