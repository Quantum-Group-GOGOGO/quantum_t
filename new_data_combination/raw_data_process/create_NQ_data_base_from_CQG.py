import pandas as pd
from env import *
import re
import tqdm
# 这个脚本理论上只要运行一次，它会从历史数据的txt中读取数据并转化成pkl的数据保存，这个pkl的数据是raw NQ数据的数据库源版本
def handling_CQG_TXT(number):
    global data_base,live_data_base
    year,season=int_to_yearseason(number)
    string=format_contract(year,season)
    CQG_data_path=data_base+'/raw/NQ_historic/Individual/NQ/NQ'+string+'.txt'
    NQ_type0_path=live_data_base+'/type0/NQ/NQBASE'+string+'.pkl'
    cols = ['date','time', 'open', 'high', 'low', 'close', 'volume']

    df = pd.read_csv(
        CQG_data_path,           # 或者你的文件路径
        header=None,          # 表示文件中没有表头行
        names=cols           # 给每列命名
    )
    #按照读入的字符串格式构造datetime时间结构
    dt_str = df['date'].astype(str) + df['time'].astype(str).str.zfill(4)
    df['datetime'] = pd.to_datetime(dt_str, format='%Y%m%d%H%M')
    #去掉原来的列，并把datetime做成索引
    df = df.set_index('datetime').drop(columns=['date', 'time'])
    #+1小时，时区换算
    df.index = df.index + pd.Timedelta(hours=1)
    df=fill_missing_minutes(df)
    df.to_pickle(NQ_type0_path)

# 查看前几行，确认类型

def fill_missing_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 DataFrame（以 DatetimeIndex 且按升序排列）中：
    - 对相邻时间差大于1分钟且小于4小时的区间，插入缺失的分钟行
    - 填充值：open/high/low/close 全用前一行的 close，volume=0
    返回补齐后的新 DataFrame。
    """
    rows = []
    timestamps = df.index

    for i in range(len(timestamps) - 1):
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

#给不同的合同编号，2000年H合同开始计数为0号合同，M为1，以此类推
def int_to_yearseason(number):
    year=2000+(number//4)
    season=number%4
    return year,season
#反函数，用于求得特定的合同编号
def yearseason_to_int(year,season):
    number=(year-2000)*4+season
    return number

#合同名字字符串换算成季节和年份的int值
def parse_contract(contract: str) -> (int, int):
    """
    解析期货合约代码，返回 (年份, 季节序号)
    合约代码格式示例：2021H, 2023M, 2024U, 2025Z
    对应季节映射：H->0, M->1, U->2, Z->3
    """
    # 季节字母到序号的映射
    season_map = {'H': 0, 'M': 1, 'U': 2, 'Z': 3}

    match = re.fullmatch(r'(\d{4})([HMUZ])', contract)
    if not match:
        raise ValueError(f"合约代码格式不正确: {contract}")

    year = int(match.group(1))
    season_letter = match.group(2)
    season = season_map[season_letter]

    return year,season

#上面函数的反函数，给定int的年份季节，输出字符串
def format_contract(year: int, season: int) -> str:
    """
    根据年份和季节序号生成期货合约代码。
    
    参数：
    - year: 4 位年份，如 2021
    - season: 季节序号，0->H, 1->M, 2->U, 3->Z
    
    返回值：
    - 合约代码字符串，例如 "2021H"
    
    抛出：
    - ValueError: 当 season 不在 [0,1,2,3] 时
    """
    season_map = {0: 'H', 1: 'M', 2: 'U', 3: 'Z'}
    
    if season not in season_map:
        raise ValueError(f"无效的季节序号: {season}，应为 0, 1, 2 或 3")
    
    return f"{year}{season_map[season]}"

year,season=parse_contract('2025U')
print(yearseason_to_int(year,season))
for i in range(yearseason_to_int(year,season)):
    handling_CQG_TXT(i)


#print(df.index.dtype)
#df.to_pickle(NQ_type0_path)