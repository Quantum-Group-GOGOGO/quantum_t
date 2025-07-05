import pandas as pd
from env import *
from tqdm import tqdm

def fill_missing_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 DataFrame（以 DatetimeIndex 且按升序排列）中：
    - 对相邻时间差大于1分钟且小于4小时的区间，插入缺失的分钟行
    - 填充值：open/high/low/close 全用前一行的 close，volume=0
    返回补齐后的新 DataFrame。
    """
    rows = []
    timestamps = df.index

    for i in tqdm(range(len(timestamps) - 1)):
        t0 = timestamps[i]
        t1 = timestamps[i + 1]
        row0 = df.loc[t0]
        # 保留原始行
        rows.append(pd.DataFrame([row0.values], index=[t0], columns=df.columns))

        # 如果缺失区间在 (1 分钟, 0.5 小时) 内，插入所有缺失分钟
        delta = t1 - t0
        if pd.Timedelta(minutes=1) < delta < pd.Timedelta(minutes=30):
            missing_times = pd.date_range(
                start=t0 + pd.Timedelta(minutes=1),
                end=t1 - pd.Timedelta(minutes=1),
                freq='min'
            )
            fill_values = {
                'open': row0['close'],
                'high': row0['close'],
                'low': row0['close'],
                'close': row0['close'],
                'volume': 0
            }
            fill_df = pd.DataFrame([fill_values] * len(missing_times), index=missing_times)
            rows.append(fill_df)

    # 添加最后一行
    last_ts = timestamps[-1]
    last_row = df.loc[last_ts]
    rows.append(pd.DataFrame([last_row.values], index=[last_ts], columns=df.columns))

    # 合并并排序
    filled_df = pd.concat(rows).sort_index()
    return filled_df

QQQBASE =pd.read_pickle(live_data_base+'/type0/QQQ/QQQ_BASE.pkl')
print(len(QQQBASE))

def is_strictly_increasing(df):
    idx = df.index
    # is_monotonic_increasing 检查的是非递减 (>=)
    # 再加上 is_unique 保证不存在相等项，就等于严格递增
    return idx.is_monotonic_increasing and idx.is_unique

print(is_strictly_increasing(QQQBASE))
QQQBASE1 = fill_missing_minutes(QQQBASE)
print(len(QQQBASE1))
QQQBASE1.to_pickle(live_data_base+'/type0/QQQ/QQQ_BASE.pkl')