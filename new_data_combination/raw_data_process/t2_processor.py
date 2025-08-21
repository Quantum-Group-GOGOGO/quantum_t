
import recording_time_trigger as rtt
from env import *
import pandas as pd
from preallocdataframe import PreallocDataFrame
from t3_processor import live_t3
from rollingmean import RollingMean
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time as tm
import os
import time

def yearseason_to_lasttime(year,season):
    month=(season+1)*3
    date=rtt.second_friday(year,month)
    return datetime(year, month, date.day, 20, 0, 0)

def yearseason_to_int(year,season):
        number=(year-2000)*4+season
        return number

def int_to_yearseason(number):
        year=2000+(number//4)
        season=number%4
        return year,season

def calculate_current_contract_year_season(now):
        if now.month==1 or now.month==2:
            return now.year,0
        elif now.month==4 or now.month==5:
            return now.year,1
        elif now.month==7 or now.month==8:
            return now.year,2
        elif now.month==10 or now.month==11:
            return now.year,3
        else:
            season=(now.month//3)-1
            if rtt.is_trigger_day_pass():
                season=season+1
            if season>3:
                return now.year+1,0
            else:
                return now.year,season
    
def calculate_current_using_contract_year_season(now):
        if now.month==1 or now.month==2:
            return now.year,0
        elif now.month==4 or now.month==5:
            return now.year,1
        elif now.month==7 or now.month==8:
            return now.year,2
        elif now.month==10 or now.month==11:
            return now.year,3
        else:
            season=(now.month//3)-1
            if rtt.is_trigger_day2_pass():
                season=season+1
            if season>3:
                return now.year+1,0
            else:
                return now.year,season
            
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

def is_index_strictly_increasing(df) -> int:
    """
    检查 DataFrame 的索引（必须为 DatetimeIndex）是否严格单调递增且无重复值。

    参数
    ----
    df : pd.DataFrame
        索引需为 DatetimeIndex。

    返回
    ----
    int
        如果索引严格递增且无重复，返回 1；否则返回 0。
    """
    idx = df.index
    # is_monotonic_increasing 保证 idx[i] ≥ idx[i-1]
    # is_unique 保证没有重复
    return int(idx.is_monotonic_increasing and idx.is_unique)

def previous_break(df: pd.DataFrame,
                   time_point: pd.Timestamp,
                   threshold: pd.Timedelta = pd.Timedelta('30min')
                  ) -> pd.Timestamp | None:
    idx = df.index
    diffs = idx.to_series().diff()
    break_positions = np.where(diffs > threshold)[0]

    # 如果 time_point 落在某个断点间隙里，直接返回该断点前时间
    for pos in break_positions:
        if idx[pos-1] < time_point < idx[pos]:
            return idx[pos-1]

    # 否则返回最后一个断点前的时间
    valid = [pos for pos in break_positions if idx[pos] < time_point]
    if not valid:
        return None

    pos = max(valid)
    return idx[pos-1]

def previous_break2(df: pd.DataFrame,
                   time_point: pd.Timestamp,
                   threshold: pd.Timedelta = pd.Timedelta('30min')
                  ) -> pd.Timestamp | None:
    idx = df.index
    diffs = idx.to_series().diff()
    break_positions = np.where(diffs > threshold)[0]

    # 如果 time_point 落在某个断点间隙里，直接返回该断点前时间
    for pos in break_positions:
        if idx[pos-1] < time_point < idx[pos]:
            return idx[pos-1]

    # 否则返回最后一个断点前的时间
    valid = [pos for pos in break_positions if idx[pos] < time_point]
    if not valid:
        return None

    pos = max(valid)
    return idx[pos]

def is_continuous_hours(df: pd.DataFrame,
                        time_point: pd.Timestamp,
                        hours: int = 4
                       ) -> int:
    """
    检查在 df（升序 datetime 索引）中，以 time_point（或它前的最近点）为终点，
    往前数 hours 小时（即 hours*60 条记录）时，是否都是严格的 1 分钟间隔。
    返回 1（连续）或 0（不连续或数据不足）。
    """
    idx = df.index
    # 找到 time_point 在 idx 中的插入位置
    pos = idx.searchsorted(time_point, side="left")

    # 如果正好等于 time_point，就从它开始；否则从前一个有效位置开始
    if pos < len(idx) and idx[pos] == time_point:
        end_pos = pos
    else:
        end_pos = pos - 1
        if end_pos < 0:
            return 0  # 没有比 time_point 更早的点

    window_size = hours * 60  # 要检查的记录条数
    # 确保有足够的点可取
    if end_pos + 1 < window_size:
        return 0

    # 取出这 window_size 个时间戳
    window = idx[end_pos - window_size + 1 : end_pos + 1]
    # 计算相邻时间差
    diffs = pd.Series(window).diff().dropna()
    # 检查是否全部等于 1 分钟
    return int((diffs == pd.Timedelta(minutes=1)).all())


def calculate_t1t2(df: pd.DataFrame,
                 time_point: pd.Timestamp,
                 hours: int = 1,
                 threshold: pd.Timedelta = pd.Timedelta('30min')
                ) -> pd.Timestamp | None:
    """
    在 df（升序 datetime 索引）中：
      从 time_point 开始，反复找上一个 “断点前时间戳” t，
      并检查以 t 为终点往前 hours 小时是否连续。
    返回第一个满足连续 hours 小时的断点前时间戳，找不到则返回 None。
    """
    t = previous_break(df, time_point, threshold=threshold)
    while t is not None:
        if is_continuous_hours(df, t, hours=hours):
            return t
        # 不满足则再往前找一个断点
        t = previous_break(df, t, threshold=threshold)
    return None

def sum_volume_between(df: pd.DataFrame,
                       t1: pd.Timestamp,
                       t2: pd.Timestamp,
                       col: str
                      ) -> float:
    """
    计算 df['volume'] 在索引介于 t1 与 t2 之间（含两端）行的总和。

    参数
    ----
    df : pd.DataFrame
        必须以 DatetimeIndex 且已升序排列，且含有名为 'volume' 的列。
    t1 : pd.Timestamp
        起始时间戳（较早）。
    t2 : pd.Timestamp
        结束时间戳（较晚）。

    返回
    ----
    float
        df.loc[t1:t2, 'volume'] 的求和结果。
    """
    # 保证 t1 <= t2
    if t1 > t2:
        t1, t2 = t2, t1

    # 用 .loc 切片并求和
    return df.loc[t1:t2, col].sum()

def mean_price_before(df, t, col, n=60):
    """
    取 df（已按 datetime 索引升序，含 'close' 列）中，
    在时间点 t 之前（或等于 t）往前数 n 行的 'close' 列均值。

    参数
    ----
    df : pd.DataFrame
        索引为 DatetimeIndex 且已升序排列，必须包含 'close' 列
    t : pd.Timestamp
        截止时间点，保证为 datetime 类型
    n : int
        往前取多少行，默认为 60

    返回
    ----
    与 df['close'] 相同的标量类型（如 float）
    """
    # 在索引中定位：找第一个 > t 的位置，然后取它前一行
    pos = df.index.searchsorted(t, side='right') - 1

    # 如果前一行不存在，或不够 n 行，则抛错
    if pos < 0 or pos + 1 < n:
        raise ValueError(f"时间 {t} 之前不足 {n} 条记录")

    # 切出那 n 行并求平均
    window = df.iloc[pos - n + 1 : pos + 1][col]
    return window.mean()

def get_ohlcv_at(df, t):
    """
    返回 df 在时间戳 t 这一行的 open, high, low, close, volume，
    按顺序作为五个变量返回。

    要求：
      - df.index 是 DatetimeIndex
      - t 是 pd.Timestamp，且恰好在 df.index 中
    """
    # 直接逐列取值
    o = df.at[t, 'open']
    h = df.at[t, 'high']
    l = df.at[t, 'low']
    c = df.at[t, 'close']
    v = df.at[t, 'volume']
    return o, h, l, c, v

def add_continue_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    给 df 增加一列 'continue'：
      - 对于第 i 行，只有当它及其前 29 行一共 30 个点都是严格 1 分钟连续时，continue=1；
      - 对于前 30 行（i < 30）或有任何断点的行，continue=0。

    要求：
      - df.index 是 DatetimeIndex，且已升序排列。
    """
    # 1. 计算相邻行的时间差，得到一个 Series（第 0 行为 NaT）
    diffs = df.index.to_series().diff()

    # 2. 标记哪些差分正好是 1 分钟
    is_one_min = (diffs == pd.Timedelta(minutes=1)).astype(int)

    # 3. 对这个 0/1 序列做长度为 30 的滚动求和，要求窗口内全是 1
    #    rolling 结果在第 i 行表示 [i-29, …, i] 这 30 个差分是否都为 1
    window_sum = is_one_min.rolling(window=60, min_periods=60).sum()

    # 4. 只有窗口和恰好为 30 才算连续
    df = df.copy()
    df['continue'] = (window_sum == 60).astype(int)

    return df

def last_continue_before(df: pd.DataFrame, t: pd.Timestamp) -> pd.Timestamp | None:
    """
    在 df（DatetimeIndex 且升序，含 'continue' 列）中，
    找到小于 t 且 continue==1 的最大索引时间戳。

    参数
    ----
    df : pd.DataFrame
        必须以 DatetimeIndex 且已升序排列，且含有 'continue' 列（0/1）。
    t : pd.Timestamp
        参考时间点。

    返回
    ----
    pd.Timestamp 或 None
        如果存在满足条件的时间点，返回那个时间戳；否则返回 None。
    """
    # 先取得所有 continue==1 的索引
    valid_idx = df.index[df['continue'] == 1]
    if len(valid_idx) == 0:
        return None

    # 在这些索引中找第一个 ≥ t 的位置，再往前一个就是 < t 的最大值
    pos = valid_idx.searchsorted(t, side='left') - 1
    if pos >= 0:
        return valid_idx[pos]
    else:
        return None
    



def sliceQQQ(A,B,buffer_lines):#A=QQQ,B=NQ
    b_start = B.index[0]
    b_end   = B.index[-1]
    pos_start = A.index.searchsorted(b_start, side="left")
    pos0 = max(0, pos_start - buffer_lines)
    new_start = A.index[pos0]
    return A.loc[new_start : b_end+pd.Timedelta(minutes=480)]

def slice8months(A):
    last_ts = A.index[-1]                       
    start_ts = last_ts - pd.Timedelta(days=133) 
    return A.loc[start_ts : last_ts]

def slice7months(A):
    last_ts = A.index[-1]                       
    start_ts = last_ts - pd.Timedelta(days=113) 
    return A.loc[start_ts : last_ts]

def slice5months(A):
    last_ts = A.index[-1]                       
    start_ts = last_ts - pd.Timedelta(days=40) 
    return A.loc[start_ts : last_ts]

def slice_remove_first_15_days(df):
    """
    如果 df 的时间跨度 > 15 天，则去掉最开始的 15 天的数据，返回剩余部分。
    索引必须为 DatetimeIndex 且升序排列。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("索引必须为 DatetimeIndex")
    
    first_ts = df.index[0]
    last_ts  = df.index[-1]
    span_days = (last_ts - first_ts).days
    
    if span_days <= 15:
        raise ValueError(f"数据跨度只有 {span_days} 天，不足 16 天，无法去除前 15 天")
    
    # 计算新的起点：第一个时间 + 15 天
    cut_ts = first_ts + pd.Timedelta(days=15)
    # 如果恰好没有这个标签，loc 会自动取 ≥ cut_ts 的第一行
    return df.loc[cut_ts:]

def slice_A_from_B_minus_20d(A: pd.DataFrame,
                            B: pd.DataFrame) -> pd.DataFrame:
    if B.empty:
        raise ValueError("B 的索引不能为空，无法计算裁剪起点")
    # 1. 取 B 的最后一个时间戳
    last_B = B.index[-1]
    # 2. 往前 8 天得到裁剪起点 t1
    t1 = last_B - pd.Timedelta(days=20)
    # 3. 用 .loc 切片，保留索引 ≥ t1 的行
    return A.loc[t1:]

def printht(df, n=5):
    top = df.head(n)
    bot = df.tail(n)
    # 用一个空行或自定义行来分隔
    sep = pd.DataFrame([ ['...'] * df.shape[1] ], columns=df.columns, index=['...'])
    snippet = pd.concat([top, sep, bot])
    print(snippet)



class live_t2:
    def __init__(self):
        self.t2_file=live_data_base+'/type2/type2Base.pkl'
        self.t0_QQQ_file=live_data_base+'/type0/QQQ/QQQ_BASE.pkl'
        self.T2Base=PreallocDataFrame(pd.read_pickle(self.t2_file))
        
        self.T2Base=self.T2Base.head(800000)
        self.QQQT0=PreallocDataFrame(pd.read_pickle(self.t0_QQQ_file))
        self.initial_dataBase()

    def link_t2t0sub(self, nqt0_processor, qqqt0_processor):
        self.nqt0_p=nqt0_processor
        self.qqqt0_p=qqqt0_processor
        self.nqt0_p.link_t2obj(self)
        self.qqqt0_p.link_t2obj(self)
        self.NQT0C=self.nqt0_p.current_contract_data
        self.NQT0N=self.nqt0_p.next_contract_data
        self.QQQT0=self.qqqt0_p.QQQBASE

    def link_t3obj(self, t3_processor:live_t3):#只允许被link_sub函数调用
        self.t3_p=t3_processor

    def initial_dataBase(self):
        last_ts = self.T2Base.index[-1]
        last_year,last_season=calculate_current_contract_year_season(last_ts)
        last_num=yearseason_to_int(last_year,last_season)
        now_year,now_season=calculate_current_contract_year_season(datetime.now())
        self.now_num=yearseason_to_int(now_year,now_season)
        #self.NQt1MeanCalculator=RollingMean()
        #self.QQQt1MeanCalculator=RollingMean()
        #self.NQt2MeanCalculator=RollingMean()
        #self.QQQt2MeanCalculator=RollingMean()
        buffers = {}
        print(f'数据库中的T2数据截止到 {last_ts} 季')
        print(f'需要拼凑出从 {last_num} 季到 {self.now_num} 季的数据')
        for i in range(last_num, self.now_num+2):
            ny,ns=int_to_yearseason(i)
            contract_str=format_contract(ny,ns)
            t0_NQ_file=live_data_base+'/type0/NQ/'+'NQBASE'+contract_str+'.pkl'
            t1_NQ_file=live_data_base+'/type1/'+'NQBaseQQQAlign'+contract_str+'.pkl'
            print(f'T2处理器正在处理 {ny} 年, {ns} 季的合约,对应总第 {i} 季')
            if not os.path.exists(t1_NQ_file):
                print('合约T1不存在，需要手动计算对齐文件')
                NQT0=PreallocDataFrame(pd.read_pickle(t0_NQ_file))
                if i==self.now_num+1:
                    NQT0=slice5months(NQT0)
                else:
                    NQT0=slice8months(NQT0)
                self.QQQT0Buffer=sliceQQQ(self.QQQT0,NQT0,0)
                if i==self.now_num+1:
                    NQT0P,self.merged_N,self.t2N,self.t4N,self.volumeRN=self.processNQ(NQT0,self.QQQT0Buffer)
                elif i==self.now_num:
                    NQT0P,self.merged_C,self.t2C,self.t4C,self.volumeRC=self.processNQ(NQT0,self.QQQT0Buffer)
                else:
                    NQT0P,_,_,_,_=self.processNQ(NQT0,self.QQQT0Buffer)
                self.QQQT0P=slice7months(self.QQQT0Buffer)
                merged = NQT0P.combine_first(self.QQQT0P)
                merged = merged.sort_index()
                merged.to_pickle(t1_NQ_file)
                buffers[i] = merged
                if i==self.now_num:
                    self.T1C=merged
                elif i==self.now_num+1 :
                    self.T1N=merged

            elif i==self.now_num:
                print('最新一期合约，硬盘状态可能不完整,正在同步T1数据')
                NQT1=PreallocDataFrame(pd.read_pickle(t1_NQ_file))
                NQT0=PreallocDataFrame(pd.read_pickle(t0_NQ_file))
                NQT0=slice_A_from_B_minus_20d(NQT0,NQT1)
                self.QQQT0Buffer=sliceQQQ(self.QQQT0,NQT0,0)
                NQT0P,self.merged_C,self.t2C,self.t4C,self.volumeRC=self.processNQ(NQT0,self.QQQT0Buffer)
                self.QQQT0P=slice_remove_first_15_days(self.QQQT0Buffer)
                merged = NQT0P.combine_first(self.QQQT0P)
                
                merged = merged.sort_index()
                NQT1.concat_small(merged)
                NQT1 = NQT1.drop_index_duplicates(keep='first')
                NQT1.sort_index()
                NQT1.to_pickle(t1_NQ_file)
                buffers[i] = NQT1
                self.T1C=NQT1
            elif i==self.now_num+1 :
                print('最新下期合约，硬盘状态可能不完整,正在同步T1数据')
                NQT1=PreallocDataFrame(pd.read_pickle(t1_NQ_file))
                NQT0=PreallocDataFrame(pd.read_pickle(t0_NQ_file))
                NQT0=slice_A_from_B_minus_20d(NQT0,NQT1)
                self.QQQT0Buffer=sliceQQQ(self.QQQT0,NQT0,0)
                NQT0P,self.merged_N,self.t2N,self.t4N,self.volumeRN=self.processNQ(NQT0,self.QQQT0Buffer)
                self.QQQT0P=slice_remove_first_15_days(self.QQQT0Buffer)
                merged = NQT0P.combine_first(self.QQQT0P)
                merged = merged.sort_index()
                NQT1.concat_small(merged)
                NQT1 = NQT1.drop_index_duplicates(keep='first')
                NQT1.sort_index()
                NQT1.to_pickle(t1_NQ_file)
                buffers[i] = NQT1
                self.T1N=NQT1
            elif i==self.now_num-1 :
                print('最新前1期合约，硬盘状态可能不完整,正在同步T1数据')
                NQT1=PreallocDataFrame(pd.read_pickle(t1_NQ_file))
                NQT0=PreallocDataFrame(pd.read_pickle(t0_NQ_file))
                NQT0=slice_A_from_B_minus_20d(NQT0,NQT1)
                self.QQQT0Buffer=sliceQQQ(self.QQQT0,NQT0,0)
                NQT0P,_,_,_,_=self.processNQ(NQT0,self.QQQT0Buffer)
                self.QQQT0P=slice_remove_first_15_days(self.QQQT0Buffer)
                merged = NQT0P.combine_first(self.QQQT0P)
                merged = merged.sort_index()
                NQT1.concat_small(merged)
                NQT1 = NQT1.drop_index_duplicates(keep='first')
                NQT1.sort_index()
                NQT1.to_pickle(t1_NQ_file)
                buffers[i] = NQT1
                
            else:
                print('老旧合约T1存在，直接读入对齐文件')
                merged=PreallocDataFrame(pd.read_pickle(t1_NQ_file))
                buffers[i] = merged

        last_year,last_season=calculate_current_using_contract_year_season(last_ts)
        last_num=yearseason_to_int(last_year,last_season)
        now_using_year,now_using_season=calculate_current_using_contract_year_season(datetime.now())
        if now_using_year!=now_year or now_using_season!=now_season:
            self.leap=1
        else:
            self.leap=0

        self.now_num=yearseason_to_int(now_using_year,now_using_season)

        for i in range(last_num, self.now_num+1):
            ny1,ns1=int_to_yearseason(i-1)
            lasttime1=yearseason_to_lasttime(ny1,ns1)
            ny2,ns2=int_to_yearseason(i)
            lasttime2=yearseason_to_lasttime(ny2,ns2)
            segment = buffers[i].loc[(buffers[i].index > lasttime1) & (buffers[i].index <= lasttime2)]
            
            if not is_index_strictly_increasing(self.T2Base):
                print('警告，self.T2Base不单调递增，正在排序，需要检查处理过程')
                self.T2Base = self.T2Base.sort_index()
                self.T2Base = self.T2Base.drop_index_duplicates()

            if not is_index_strictly_increasing(segment):
                print('警告，新T1合约不单调递增，正在排序，需要检查处理过程')
                segment = segment.sort_index()
                segment = segment[~segment.index.duplicated(keep='first')]

            segment = segment.loc[(segment.index > last_ts) & (segment.index <= lasttime2)]
            print(f'准备拼接第 {ny2} 年, {ns2} 季的合约,对应总第 {i} 季')
            #print('拼接前的T2数据末尾')
            #print(self.T2Base.tail())
            #print('拼接前准备好的数据')
            #printht(segment)
            self.T2Base.concat_small(segment)
            if not is_index_strictly_increasing(self.T2Base):
                print('警告，self.T2Base在拼接后不单调递增，正在排序，需要检查处理过程')
                self.T2Base = self.T2Base.sort_index()
                self.T2Base = self.T2Base.drop_index_duplicates()
            else:
                print('T2Base拼接完成，无检测异常')

        print('正在保存T2Base')        
        self.T2Base.to_pickle(self.t2_file)
        print('T2Base同步已完成') 
    
    def live_change_using(self):
        self.leap=True

    def calculate_using_contract(self,time):
        return yearseason_to_int(*calculate_current_using_contract_year_season(time))

    def calculate_contract(self,time):
        return yearseason_to_int(*calculate_current_contract_year_season(time))      
    
    def calculate_price_ratio12(self,df: pd.DataFrame,
                            t1: pd.Timestamp,
                            t2: pd.Timestamp,
                            colNQ: str,
                            colQQQ: str):
        NQprice1=mean_price_before(df, t1, colNQ)
        QQQprice1=mean_price_before(df, t1, colQQQ)
        #NQprice1=self.NQt1MeanCalculator.mean_before(df, t1, colNQ)
        #QQQprice1=self.QQQt1MeanCalculator.mean_before(df, t1, colNQ)
        ratio1=NQprice1/QQQprice1
        NQprice2=mean_price_before(df, t2, colNQ)
        QQQprice2=mean_price_before(df, t2, colQQQ)
        #NQprice2=self.NQt2MeanCalculator.mean_before(df, t1, colNQ)
        #QQQprice2=self.QQQt2MeanCalculator.mean_before(df, t1, colNQ)
        ratio2=NQprice2/QQQprice2
        return ratio1,ratio2

    def processNQ(self,NQT0,QQQT0):

        merged_df = NQT0.join(QQQT0,how='inner',lsuffix='_NQ',rsuffix='_QQQ')
        merged_df=add_continue_flag(merged_df)

        NQT0=slice_remove_first_15_days(NQT0)
        QQQT0=sliceQQQ(QQQT0,NQT0,0)

        times = NQT0.index
        df2 = NQT0.copy()
        
        df2['open']   = np.nan
        df2['high']   = np.nan
        df2['low']    = np.nan
        df2['close']  = np.nan
        df2['volume'] = np.nan
        changes=1
        prev_time = times[0] - pd.Timedelta(days=1)
        for time  in tqdm(times, desc="Processing timestamps"):
            delta = time - prev_time
            if delta > pd.Timedelta(minutes=1):
                t1=calculate_t1t2(merged_df,time)
                
                print(merged_df.head())
                print(merged_df.tail())
                print(t1)
                print(time)
                t2=calculate_t1t2(merged_df,t1)
                t3=previous_break2(merged_df,t1)
                NQvol=sum_volume_between(merged_df,t1,t3,'volume_NQ')
                QQQvol=sum_volume_between(merged_df,t1,t3,'volume_QQQ')

                

                volume_ratio=NQvol/QQQvol
            t4 = last_continue_before(merged_df,time)
            open_ratio4,open_ratio2=self.calculate_price_ratio12(merged_df,t4,t2,'open_NQ','open_QQQ')
            high_ratio4,high_ratio2=self.calculate_price_ratio12(merged_df,t4,t2,'high_NQ','high_QQQ')
            low_ratio4,low_ratio2=self.calculate_price_ratio12(merged_df,t4,t2,'low_NQ','low_QQQ')
            close_ratio4,close_ratio2=self.calculate_price_ratio12(merged_df,t4,t2,'close_NQ','close_QQQ')
            ratio4=(open_ratio4+high_ratio4+low_ratio4+close_ratio4)/4
            ratio2=(open_ratio2+high_ratio2+low_ratio2+close_ratio2)/4
            
            
            delta24 = t2 - t4
            seconds24 = delta24.total_seconds()
            deltatime4 = time - t4
            secondsx1 = deltatime4.total_seconds()+1800
            price_ratio=ratio4+((ratio2-ratio4)*(secondsx1)/(seconds24))

                #o1,h1,l1,c1,v1=get_ohlcv_at(NQT0, time)

            df2.at[time, 'open']   = NQT0.at[time, 'open']   / price_ratio
            df2.at[time, 'high']   = NQT0.at[time, 'high']   / price_ratio
            df2.at[time, 'low']    = NQT0.at[time, 'low']    / price_ratio
            df2.at[time, 'close']  = NQT0.at[time, 'close']  / price_ratio
            df2.at[time, 'volume'] = NQT0.at[time, 'volume'] / volume_ratio
            prev_time=time
            #time = datetime(2024, 11, 1, 15, 31, 0)
            
            # o3=o1/open_ratio
            # h3=h1/high_ratio
            # l3=l1/low_ratio
            # c3=c1/close_ratio
            # v3=v1/volume_ratio
            

        return df2,PreallocDataFrame(merged_df),t2,t4,volume_ratio

    def slow_march(self):

        print('正在保存T2Base')        
        self.T2Base.to_pickle(self.t2_file)
        print('T2Base同步已完成') 
        self.initial_dataBase()
        self.t3_p.liveInitial()

    async def multi_fast_march(self,contract_,NQstatus):
        if contract_ ==1:
            delta=self.NQT0C.index[-1]-self.T2Base.index[-1]
        elif contract_==2:
            delta=self.NQT0N.index[-1]-self.T2Base.index[-1]
        elif contract_==0:
            delta=self.QQQT0.index[-1]-self.T2Base.index[-1]
        minute = int(delta.total_seconds() / 60)

        for m in range(minute-1, -1, -1):
            

            if contract_==1:
                datetime_=self.NQT0C.index[-(m+1)]
                open_=self.NQT0C['open'].iloc[-(m+1)]
                high_=self.NQT0C['high'].iloc[-(m+1)]
                low_=self.NQT0C['low'].iloc[-(m+1)]
                close_=self.NQT0C['close'].iloc[-(m+1)]
                volume_=self.NQT0C['volume'].iloc[-(m+1)]
                end= None if m==0 else -m
                idx1 = pd.Index([self.NQT0C.index[-(2+m)]])    # 长度 1
                
                idx2 = self.QQQT0.index[-(5+m):end]
                intersect_idx = idx1.intersection(idx2)

                #intersect_idx = self.NQT0C.index[-2].intersection(self.QQQT0.index[-5:])
                if not intersect_idx.empty:
                    C_new = self.NQT0C.loc[intersect_idx].join(self.QQQT0.loc[intersect_idx], how='inner',lsuffix='_NQ', rsuffix='_QQQ')
                    continue_now = True if (C_new.index[-1]-self.merged_C.index[-1]==pd.Timedelta(minutes=1)) else False
                    if self.merged_C['continue'].iloc[-1]==1:
                        if continue_now:
                            #如果前一行连续，且当前行和前一行的差值也正好是1分钟
                            C_new['continue'] = 1
                        else:
                            #如果前一行连续，但当前行和前面差别很大
                            C_new['continue'] = 0
                    else:
                        last59 = self.merged_C.index[-59:]
                        ns = last59.asi8
                        deltas = ns[1:] - ns[:-1]
                        one_min_ns = np.timedelta64(1, 'm').astype('timedelta64[ns]')
                        if continue_now and np.all(deltas == one_min_ns):
                            #如果前一行不连续，但当前行和前面59行差别正好都是1分钟
                            C_new['continue'] = 1
                        else:
                            #否则延续不连续记录
                            C_new['continue'] = 0
                    self.merged_C.concat_small(C_new)
                #完成对merge_df的更新

                if datetime_-self.NQT0C.index[-2-m] > pd.Timedelta(minutes=1):#如果NQT0数据是不连续的，需要重新对齐t2和volume
                    t1=calculate_t1t2(self.merged_C,datetime_)
                    self.t2C=calculate_t1t2(self.merged_C,t1)
                    t3=previous_break2(self.merged_C,t1)
                    NQvol=sum_volume_between(self.merged_C,t1,t3,'volume_NQ')
                    QQQvol=sum_volume_between(self.merged_C,t1,t3,'volume_QQQ')
                    self.volumeRC=NQvol/QQQvol


                t4 = last_continue_before(self.merged_C,datetime_)

                open_ratio4,open_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'open_NQ','open_QQQ')
                high_ratio4,high_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'high_NQ','high_QQQ')
                low_ratio4,low_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'low_NQ','low_QQQ')
                close_ratio4,close_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'close_NQ','close_QQQ')

                ratio4=(open_ratio4+high_ratio4+low_ratio4+close_ratio4)/4
                ratio2=(open_ratio2+high_ratio2+low_ratio2+close_ratio2)/4


                delta24 = self.t2C - t4
                seconds24 = delta24.total_seconds()
                deltatime4 = datetime_ - t4
                secondsx1 = deltatime4.total_seconds()+1800
                price_ratio=ratio4+((ratio2-ratio4)*(secondsx1)/(seconds24))

                o=open_/price_ratio
                h=high_/price_ratio
                l=low_/price_ratio
                c=close_/price_ratio
                v=volume_/self.volumeRC

                self.T1C.insert_row_keep_last(datetime_, [o, h, l, c, v])
                print(f'{datetime.now()}  NQ当前T1不连续数据处理，添加：{datetime_} {o} {h} {l} {c} {v}')


                if self.leap==0:
                    self.T2Base.insert_row_keep_last(datetime_, [o, h, l, c, v])
                    print(f'{datetime.now()}  NQ当前活跃，T2采纳NQ当前数据：{datetime_} {o} {h} {l} {c} {v}')
                    await self.t3_p.liveRenew(1)
                else:
                    await self.t3_p.liveRenew(0)
                        
            elif contract_==2:

                datetime_=self.NQT0N.index[-(m+1)]
                open_=self.NQT0N['open'].iloc[-(m+1)]
                high_=self.NQT0N['high'].iloc[-(m+1)]
                low_=self.NQT0N['low'].iloc[-(m+1)]
                close_=self.NQT0N['close'].iloc[-(m+1)]
                volume_=self.NQT0N['volume'].iloc[-(m+1)]
                end= None if m==0 else -m
                idx1 = pd.Index([self.NQT0N.index[-2-m]])    # 长度 1
                idx2 = self.QQQT0.index[-(5+m):end]
                intersect_idx = idx1.intersection(idx2)

                #intersect_idx = self.NQT0N.index[-2].intersection(self.QQQT0.index[-5:])
                if not intersect_idx.empty:
                    N_new = self.NQT0N.loc[intersect_idx].join(self.QQQT0.loc[intersect_idx], how='inner',lsuffix='_NQ', rsuffix='_QQQ')
                    continue_now = True if (N_new.index[-1]-self.merged_N.index[-1]==pd.Timedelta(minutes=1)) else False
                    if self.merged_N['continue'].iloc[-1]==1:
                        if continue_now:
                            #如果前一行连续，且当前行和前一行的差值也正好是1分钟
                            N_new['continue'] = 1
                        else:
                            #如果前一行连续，但当前行和前面差别很大
                            N_new['continue'] = 0
                    else:
                        last59 = self.merged_N.index[-59:]
                        ns = last59.asi8
                        deltas = ns[1:] - ns[:-1]
                        one_min_ns = np.timedelta64(1, 'm').astype('timedelta64[ns]')
                        if continue_now and np.all(deltas == one_min_ns):
                            #如果前一行不连续，但当前行和前面59行差别正好都是1分钟
                            N_new['continue'] = 1
                        else:
                            #否则延续不连续记录
                            N_new['continue'] = 0
                    self.merged_N.concat_small(N_new)
                #完成对merge_df的更新

                if datetime_-self.NQT0N.index[-2-m] > pd.Timedelta(minutes=1):#如果NQT0数据是不连续的，需要重新对齐t2和volume
                    t1=calculate_t1t2(self.merged_N,datetime_)
                    self.t2N=calculate_t1t2(self.merged_N,t1)
                    t3=previous_break2(self.merged_N,t1)
                    NQvol=sum_volume_between(self.merged_N,t1,t3,'volume_NQ')
                    QQQvol=sum_volume_between(self.merged_N,t1,t3,'volume_QQQ')
                    self.volumeRN=NQvol/QQQvol


                t4 = last_continue_before(self.merged_N,datetime_)

                open_ratio4,open_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'open_NQ','open_QQQ')
                high_ratio4,high_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'high_NQ','high_QQQ')
                low_ratio4,low_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'low_NQ','low_QQQ')
                close_ratio4,close_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'close_NQ','close_QQQ')

                ratio4=(open_ratio4+high_ratio4+low_ratio4+close_ratio4)/4
                ratio2=(open_ratio2+high_ratio2+low_ratio2+close_ratio2)/4


                delta24 = self.t2N - t4
                seconds24 = delta24.total_seconds()
                deltatime4 = datetime_ - t4
                secondsx1 = deltatime4.total_seconds()+1800
                price_ratio=ratio4+((ratio2-ratio4)*(secondsx1)/(seconds24))
                

                o=open_/price_ratio
                h=high_/price_ratio
                l=low_/price_ratio
                c=close_/price_ratio
                v=volume_/self.volumeRN

                self.T1N.insert_row_keep_last(datetime_, [o, h, l, c, v])
                print(f'{datetime.now()}  NQ下季T1不连续数据处理，添加：{datetime_} {o} {h} {l} {c} {v}')


                if self.leap==1:
                    self.T2Base.insert_row_keep_last(datetime_, [o, h, l, c, v])
                    print(f'{datetime.now()}  NQ下季活跃，T2采纳NQ下季数据：{datetime_} {o} {h} {l} {c} {v}')
                    await self.t3_p.liveRenew(1)
                else:
                    await self.t3_p.liveRenew(0)

                        
            elif contract_==0:
                datetime_=self.QQQT0.index[-(m+1)]
                open_=self.QQQT0['open'].iloc[-(m+1)]
                high_=self.QQQT0['high'].iloc[-(m+1)]
                low_=self.QQQT0['low'].iloc[-(m+1)]
                close_=self.QQQT0['close'].iloc[-(m+1)]
                volume_=self.QQQT0['volume'].iloc[-(m+1)]
                end= None if m==0 else -m

                self.T1C.insert_row_keep_first(datetime_, [open_, high_, low_, close_, volume_])
                self.T1N.insert_row_keep_first(datetime_, [open_, high_, low_, close_, volume_])

                print(f'{datetime.now()}  QQQ T1不连续数据处理完毕：{datetime_} {open_} {high_} {low_} {close_} {volume_}')
                if not NQstatus:
                    self.T2Base.insert_row_keep_first(datetime_, [open_, high_, low_, close_, volume_])
                    print(f'{datetime.now()}  NQ不活跃，T2采纳QQQ数据：{datetime_} {open_} {high_} {low_} {close_} {volume_}')
                    await self.t3_p.liveRenew(0)
        
        
        
        print('NQ 当前T2不连续数据添加完成')
        





    async def fast_march(self,datetime_,open_,high_,low_,close_,volume_,contract_,NQstatus):

        if contract_ ==1:
            delta=self.NQT0C.index[-1]-self.T2Base.index[-1]
        elif contract_==2:
            delta=self.NQT0N.index[-1]-self.T2Base.index[-1]
        elif contract_==0:
            delta=self.QQQT0.index[-1]-self.T2Base.index[-1]
        minute = int(delta.total_seconds() / 60)

        if minute !=1:
            print(f'Warning Minute is not 1 {minute} {delta} {contract_} {self.T2Base.index[-1]} ')
            await self.multi_fast_march(contract_,NQstatus)
            return
        else:
            if contract_==1:

                idx1 = pd.Index([self.NQT0C.index[-2]])    # 长度 1
                idx2 = self.QQQT0.index[-5:]
                intersect_idx = idx1.intersection(idx2)

                #intersect_idx = self.NQT0C.index[-2].intersection(self.QQQT0.index[-5:])
                if not intersect_idx.empty:
                    C_new = self.NQT0C.loc[intersect_idx].join(self.QQQT0.loc[intersect_idx], how='inner',lsuffix='_NQ', rsuffix='_QQQ')
                    continue_now = True if (C_new.index[-1]-self.merged_C.index[-1]==pd.Timedelta(minutes=1)) else False
                    if self.merged_C['continue'].iloc[-1]==1:
                        if continue_now:
                            #如果前一行连续，且当前行和前一行的差值也正好是1分钟
                            C_new['continue'] = 1
                        else:
                            #如果前一行连续，但当前行和前面差别很大
                            C_new['continue'] = 0
                    else:
                        last59 = self.merged_C.index[-59:]
                        ns = last59.asi8
                        deltas = ns[1:] - ns[:-1]
                        one_min_ns = np.timedelta64(1, 'm').astype('timedelta64[ns]')
                        if continue_now and np.all(deltas == one_min_ns):
                            #如果前一行不连续，但当前行和前面59行差别正好都是1分钟
                            C_new['continue'] = 1
                        else:
                            #否则延续不连续记录
                            C_new['continue'] = 0
                    self.merged_C.concat_small(C_new)
                #完成对merge_df的更新

                if datetime_-self.NQT0C.index[-2] > pd.Timedelta(minutes=1):#如果NQT0数据是不连续的，需要重新对齐t2和volume
                    t1=calculate_t1t2(self.merged_C,datetime_)
                    self.t2C=calculate_t1t2(self.merged_C,t1)
                    t3=previous_break2(self.merged_C,t1)
                    NQvol=sum_volume_between(self.merged_C,t1,t3,'volume_NQ')
                    QQQvol=sum_volume_between(self.merged_C,t1,t3,'volume_QQQ')
                    self.volumeRC=NQvol/QQQvol


                t4 = last_continue_before(self.merged_C,datetime_)

                open_ratio4,open_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'open_NQ','open_QQQ')
                high_ratio4,high_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'high_NQ','high_QQQ')
                low_ratio4,low_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'low_NQ','low_QQQ')
                close_ratio4,close_ratio2=self.calculate_price_ratio12(self.merged_C,t4,self.t2C,'close_NQ','close_QQQ')

                ratio4=(open_ratio4+high_ratio4+low_ratio4+close_ratio4)/4
                ratio2=(open_ratio2+high_ratio2+low_ratio2+close_ratio2)/4


                delta24 = self.t2C - t4
                seconds24 = delta24.total_seconds()
                deltatime4 = datetime_ - t4
                secondsx1 = deltatime4.total_seconds()+1800
                price_ratio=ratio4+((ratio2-ratio4)*(secondsx1)/(seconds24))
                
                o=open_/price_ratio
                h=high_/price_ratio
                l=low_/price_ratio
                c=close_/price_ratio
                v=volume_/self.volumeRC
                self.T1C.append_row_keep_last(datetime_, [o, h, l, c, v])
                print(f'{datetime.now()}  NQ当前T1连续数据处理完毕：{datetime_} {o} {h} {l} {c} {v}')
            
                if self.leap==0:
                    self.T2Base.append_row_keep_last(datetime_, [o, h, l, c, v])
                    print(f'{datetime.now()}  NQ当前活跃，T2采纳NQ当前数据：{datetime_} {o} {h} {l} {c} {v}')
                    await self.t3_p.liveRenew(1)
                else:
                    await self.t3_p.liveRenew(0)

                
            elif contract_==2:

                
                idx1 = pd.Index([self.NQT0N.index[-2]])    # 长度 1
                idx2 = self.QQQT0.index[-5:]
                intersect_idx = idx1.intersection(idx2)

                #intersect_idx = self.NQT0N.index[-2].intersection(self.QQQT0.index[-5:])
                if not intersect_idx.empty:
                    N_new = self.NQT0N.loc[intersect_idx].join(self.QQQT0.loc[intersect_idx], how='inner',lsuffix='_NQ', rsuffix='_QQQ')
                    continue_now = True if (N_new.index[-1]-self.merged_N.index[-1]==pd.Timedelta(minutes=1)) else False
                    if self.merged_N['continue'].iloc[-1]==1:
                        if continue_now:
                            #如果前一行连续，且当前行和前一行的差值也正好是1分钟
                            N_new['continue'] = 1
                        else:
                            #如果前一行连续，但当前行和前面差别很大
                            N_new['continue'] = 0
                    else:
                        last59 = self.merged_N.index[-59:]
                        ns = last59.asi8
                        deltas = ns[1:] - ns[:-1]
                        one_min_ns = np.timedelta64(1, 'm').astype('timedelta64[ns]')
                        if continue_now and np.all(deltas == one_min_ns):
                            #如果前一行不连续，但当前行和前面59行差别正好都是1分钟
                            N_new['continue'] = 1
                        else:
                            #否则延续不连续记录
                            N_new['continue'] = 0
                    self.merged_N.concat_small(N_new)
                #完成对merge_df的更新

                if datetime_-self.NQT0N.index[-2] > pd.Timedelta(minutes=1):#如果NQT0数据是不连续的，需要重新对齐t2和volume
                    t1=calculate_t1t2(self.merged_N,datetime_)
                    self.t2N=calculate_t1t2(self.merged_N,t1)
                    t3=previous_break2(self.merged_N,t1)
                    NQvol=sum_volume_between(self.merged_N,t1,t3,'volume_NQ')
                    QQQvol=sum_volume_between(self.merged_N,t1,t3,'volume_QQQ')
                    self.volumeRN=NQvol/QQQvol


                t4 = last_continue_before(self.merged_N,datetime_)

                open_ratio4,open_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'open_NQ','open_QQQ')
                high_ratio4,high_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'high_NQ','high_QQQ')
                low_ratio4,low_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'low_NQ','low_QQQ')
                close_ratio4,close_ratio2=self.calculate_price_ratio12(self.merged_N,t4,self.t2N,'close_NQ','close_QQQ')

                ratio4=(open_ratio4+high_ratio4+low_ratio4+close_ratio4)/4
                ratio2=(open_ratio2+high_ratio2+low_ratio2+close_ratio2)/4


                delta24 = self.t2N - t4
                seconds24 = delta24.total_seconds()
                deltatime4 = datetime_ - t4
                secondsx1 = deltatime4.total_seconds()+1800
                price_ratio=ratio4+((ratio2-ratio4)*(secondsx1)/(seconds24))
                

                o=open_/price_ratio
                h=high_/price_ratio
                l=low_/price_ratio
                c=close_/price_ratio
                v=volume_/self.volumeRN

                self.T1N.append_row_keep_last(datetime_, [o, h, l, c, v])
                

                print(f'{datetime.now()}  NQ下季T1连续数据处理完毕：{datetime_} {o} {h} {l} {c} {v}')
                if self.leap==1:
                    self.T2Base.append_row_keep_last(datetime_, [o, h, l, c, v])
                    print(f'{datetime.now()}  NQ下季活跃，T2采纳NQ下季数据：{datetime_} {o} {h} {l} {c} {v}')
                    await self.t3_p.liveRenew(1)
                else:
                    await self.t3_p.liveRenew(0)

            elif contract_==0:
                self.T1C.append_row_keep_first(datetime_, [open_, high_, low_, close_, volume_])
                self.T1N.append_row_keep_first(datetime_, [open_, high_, low_, close_, volume_])
                print(f'{datetime.now()}  QQQ T1连续数据处理完毕：{datetime_} {open_} {high_} {low_} {close_} {volume_}')
                if not NQstatus:
                        self.T2Base.append_row(datetime_, [open_, high_, low_, close_, volume_])
                        print(f'{datetime.now()}  NQ不活跃，T2采纳QQQ数据：{datetime_} {open_} {high_} {low_} {close_} {volume_}')

                await self.t3_p.liveRenew(0)
            #先把对应的current,next的T1更新完


        

#segment = df[(df.index > t1) & (df.index <= t2)]
