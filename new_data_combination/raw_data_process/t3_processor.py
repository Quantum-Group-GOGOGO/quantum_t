from env import *
import pandas as pd
import numpy as np
from datetime import datetime, time as dtime, timedelta
from preallocdataframe import PreallocDataFrame
from t6_processor import live_t6
from tqdm import tqdm


# 定义一个函数来计算 week_fraction
def calculate_week_fraction(dt):
    # 一周中的第几天（周一是 0，周日是 6）
    day_of_week = dt.weekday()  # 0 到 6
    
    # 当天已经经过的秒数
    seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    
    # 计算该时刻在本周中的秒数
    seconds_in_week_at_this_time = day_of_week * 24 * 3600 + seconds_in_day
    
    # 计算该时刻占一周的比例
    return seconds_in_week_at_this_time / 604800, seconds_in_day/86400



class live_t3:
    def __init__(self):
        self.t2FilePath=live_data_base+'/type2/type2Base.pkl'
        self.t3FilePath=live_data_base+'/type3/type3Base.pkl'
        self.eventDatePath=live_data_base+'/big_event/events.pkl'
        self.holidayDatePath=live_data_base+'/big_event/holiday.pkl'
        self.c3d=1440*3
        self.lrt=1440*7
        
        self.eventDate=PreallocDataFrame(pd.read_pickle(self.eventDatePath))
        self.eventDatetime=self.eventDate.index
        self.holidayDate=PreallocDataFrame(pd.read_pickle(self.holidayDatePath))
        self.holidayDate.index = pd.to_datetime(self.holidayDate.index)
        self.holidayDatetime=self.holidayDate.index
        self.initial_dataBase()
        print('T3初始化完成')
        
        

    def link_t3t2sub(self, t2_processor):
        self.t2_p=t2_processor
        self.t2_p.link_t3obj(self)
        self.t2Base=self.t2_p.T2Base
        self.T3Renew()

    def link_t6obj(self, t6_processor:live_t6):#只允许被link_sub函数调用
        self.t6_p=t6_processor

    def initial_dataBase(self):
        self.t2Base= PreallocDataFrame(pd.read_pickle(self.t2FilePath))
        if os.path.exists(self.t3FilePath):
            self.t3Base = PreallocDataFrame(pd.read_pickle(self.t3FilePath))
            self.close10=self.sum_last(10,'close')
            self.close60=self.sum_last(60,'close')
            self.close240=self.sum_last(240,'close')
            self.close1380=self.sum_last(1380,'close')
            self.volume10=self.sum_last(10,'volume')
            self.volume60=self.sum_last(60,'volume')
            self.volume240=self.sum_last(240,'volume')
            self.volume1380=self.sum_last(1380,'volume')
            self.volweek_raw=self.sum_last(self.lrt,'log_ret')
            self.close3d=self.sum_last(self.c3d,'close')

            self.T3Renew()
            self.t3Base.to_pickle(self.t3FilePath)

        else:
            after_df=self.processT2Base(self.t2Base)
            after_df.to_pickle(self.t3FilePath)
            self.t3Base = after_df
            self.close10=self.sum_last(10,'close')
            self.close60=self.sum_last(60,'close')
            self.close240=self.sum_last(240,'close')
            self.close1380=self.sum_last(1380,'close')
            self.volume10=self.sum_last(10,'volume')
            self.volume60=self.sum_last(60,'volume')
            self.volume240=self.sum_last(240,'volume')
            self.volume1380=self.sum_last(1380,'volume')
            self.volweek_raw=self.sum_last(self.lrt,'log_ret')
            self.close3d=self.sum_last(self.c3d,'close')
        

    def processT2Base(self,processing_dfin):
        processing_df=processing_dfin.copy()
        cols = ['event', 'pre_event', 'post_event', 'time_break_flag', 'pre_break', 'post_break']

        for c in cols:
            processing_df[c] = 0

        # 生成event 列
        common_idx = processing_df.index.intersection(self.eventDate.index)
        processing_df.loc[common_idx, 'event'] = self.eventDate.loc[common_idx, 'item']


        # 计算前一个时间点的时间差
        processing_df['time_diff'] = np.abs(processing_df.index.to_series().diff())
        # 计算后一个时间点的时间差
        processing_df['time_diff_next'] = np.abs(processing_df.index.to_series().diff(-1))

        # 生成 time_break_flag 列
        processing_df['time_break_flag'] = np.where(
            (processing_df['time_diff'] > pd.Timedelta(hours=4)) | (processing_df['time_diff_next'] > pd.Timedelta(hours=4)),
            1,
            0
        )

        # 1. 把 eventDate.index 转成一个 Series（索引和值都是事件时间）
        event_times = self.eventDate.index.to_series()

        # 2. 把 processing_df.index 也转成一个 Series
        base_times  = processing_df.index.to_series()

        # 3. 计算 pre_event：当前行时间减去最近一次（早于等于它的）事件时间
        pre = (base_times
            - event_times.reindex(base_times.index, method='ffill')
            ).dt.total_seconds() / 60

        # 4. 计算 post_event：下一个（大于等于它的）事件时间减去当前行时间
        post = (event_times.reindex(base_times.index, method='bfill')
                - base_times
            ).dt.total_seconds() / 60

        # 5. 写回 processing_df
        processing_df['pre_event']  = pre
        processing_df['post_event'] = post


        # 先拿到一个以索引为值的 Series
        dt = processing_df.index.to_series()

        # 找到 break 事件的 mask
        mask = processing_df['time_break_flag'] == 1

        # 计算 pre_break：当前时间减去最近一次 break 的时间
        processing_df['pre_break'] = (
            dt
            - dt[mask].reindex(processing_df.index, method='ffill')
        ).dt.total_seconds() / 60

        # 计算 post_break：下一次 break 的时间减去当前时间
        processing_df['post_break'] = (
            dt[mask].reindex(processing_df.index, method='bfill')
            - dt
        ).dt.total_seconds() / 60



        # 1. 找到第一个 time_break_flag==1 的时间
        first_break = processing_df.index[processing_df['time_break_flag'] == 1]
        if len(first_break) > 0:
            first_break_ts = first_break[0]

            # 2. 对所有在它之前的行，重新计算 pre_break
            mask = processing_df.index < first_break_ts
            for ts in processing_df.index[mask]:
                # 计算到最近的周日距离多少天
                # Python weekday(): Mon=0 ... Sun=6
                days_since_sunday = (ts.weekday() + 1) % 7
                prev_sunday_date = (ts.date() - timedelta(days=days_since_sunday))
                # 参考时间：周日 18:00
                ref_dt = datetime.combine(prev_sunday_date, dtime(18, 0))

                # 计算分钟差并写回 pre_break
                processing_df.at[ts, 'pre_break'] = (ts - ref_dt).total_seconds() / 60
                




        break_idxs = processing_df.index[processing_df['time_break_flag'] == 1]
        if len(break_idxs) > 0:
            last_break_ts = break_idxs[-1]

            # —— 2. 针对所有在 last_break_ts 之后的行
            mask_tail = processing_df.index > last_break_ts
            for ts in processing_df.index[mask_tail]:
                # 2.1 计算周六前一日 17:00 参考时间
                days_until_sat = (5 - ts.weekday()) % 7
                next_sat_date = ts.date() + timedelta(days=days_until_sat)
                sat_ref = datetime.combine(next_sat_date, dtime(0, 0))

                # 2.2 查找 next holiday
                future_holidays = self.holidayDate.index[self.holidayDate.index > ts]
                if len(future_holidays) > 0:
                    hol_ts = future_holidays[0]
                    # 比较哪个更近
                    delta_hol = hol_ts - ts
                    delta_sat = sat_ref - ts
                    if delta_hol < delta_sat:
                        ref_dt = hol_ts
                    else:
                        ref_dt = sat_ref
                else:
                    # 没有后续 holiday，就用 sat_ref
                    ref_dt = sat_ref
                    
                ref_dt= ref_dt - timedelta(hours=4)  # 周五 17:00

                # —— 3. 计算分钟差并写回 post_break
                processing_df.at[ts, 'post_break'] = (ref_dt - ts).total_seconds() / 60



        # 先把所有 (week_frac, day_frac) 对收集到一个列表
        fracs = list(processing_df.index.map(calculate_week_fraction))

        # 然后拆包成两列
        processing_df['week_fraction'] = [wf for wf, df_ in fracs]
        processing_df['day_fraction']  = [df_ for wf, df_ in fracs]
        #计算正弦
        # 新增一列，计算 sin(time_fraction * 2 * pi)
        processing_df['sinT'] = 0.5*(np.sin(processing_df['day_fraction'] * 2 * np.pi)+1)
        processing_df['cosT'] = 0.5*(np.cos(processing_df['day_fraction'] * 2 * np.pi)+1)

        processing_df['week_fraction_sin'] = 0.5*(np.sin(processing_df['week_fraction'] * 2 * np.pi)+1)
        processing_df['week_fraction_cos'] = 0.5*(np.cos(processing_df['week_fraction'] * 2 * np.pi)+1)

        reference_time = pd.Timestamp('2000-01-01 00:00:00')
        processing_df['absolute_time'] = (processing_df.index - reference_time).total_seconds() / 60

        processing_df.drop(columns=['time_diff', 'time_diff_next','event', 'time_break_flag','week_fraction','day_fraction',], inplace=True)


        processing_df['volume_10'] = processing_df['volume'].rolling(window=10, min_periods=1).mean()
        processing_df['volume_60'] = processing_df['volume'].rolling(window=60, min_periods=1).mean()
        processing_df['volume_240'] = processing_df['volume'].rolling(window=240, min_periods=1).mean()
        processing_df['volume_1380'] = processing_df['volume'].rolling(window=1830, min_periods=1).mean()

        processing_df['close_10'] = processing_df['close'].rolling(window=10, min_periods=1).mean()
        processing_df['close_60'] = processing_df['close'].rolling(window=60, min_periods=1).mean()
        processing_df['close_240'] = processing_df['close'].rolling(window=240, min_periods=1).mean()
        processing_df['close_1380'] = processing_df['close'].rolling(window=1830, min_periods=1).mean()
        processing_df['log_ret'] = (np.log(processing_df['close'] / processing_df['close'].shift(1)))**2
        processing_df['log_ret'] = processing_df['log_ret'].fillna(0)
        processing_df['volweek_raw'] = processing_df['log_ret'].rolling(window=self.lrt, min_periods=1).mean()
        processing_df['close3d'] = processing_df['close'].rolling(window=self.c3d, min_periods=1).mean()
        return PreallocDataFrame(processing_df)
    
    def liveInitial(self):
        self.eventDate=PreallocDataFrame(pd.read_pickle(self.eventDatePath))
        self.eventDatetime=self.eventDate.index
        self.holidayDate=PreallocDataFrame(pd.read_pickle(self.holidayDatePath))
        self.holidayDate.index = pd.to_datetime(self.holidayDate.index)
        self.holidayDatetime=self.holidayDate.index
        self.initial_dataBase()
        self.t6_p.liveInitial()

    async def liveRenew(self,decide):
        self.T3Renew()
        await self.t6_p.liveRenew(decide)

    def T3Renew(self):
        
        # 取出 t3Base 最后一行的时间
        last_t3_ts = self.t3Base.index[-1]

        # 如果要包含等于 last_t3_ts 的那一行：
        processing_df = self.t2Base.loc[last_t3_ts + pd.Timedelta('1ns'):].copy()
        
        if len(processing_df) == 0:
            if self.t3Base.index[-1] == self.t2Base.index[-1]:
                print('T3已与T2对齐，无需更新')
            else:
                print('警告，T2更新后T3无法更新，最晚时间早于T2')
                print(self.t2Base.tail())
                print(self.t3Base.tail())
                return
        
        #print(self.t3Base.columns)
        #print(last_row.columns)

        print(f'T3 {len(processing_df)} 行数据更新')

        for i, (idx, row) in enumerate(processing_df.iterrows()):
            last_row = self.t3Base.iloc[-1]
            
            post_event=last_row['post_event']-1
            post_break=last_row['post_break']-1
            pre_event=last_row['pre_event']+1
            pre_break=last_row['pre_break']+1

            log_ret = (np.log(row['close'] / last_row['close']))**2 
            
            if post_event==0:
                pre_event=1
                post_event=100000
                pos = self.eventDatetime.searchsorted(idx, side='right')
                if pos < len(self.eventDatetime):
                    next_event_time = self.eventDatetime[pos]
                    post_event=(next_event_time-idx).total_seconds() / 60

                
            
            if post_break==0:
                pre_break=1
                post_break=100000
                days_until_sat = (5 - idx.weekday()) % 7
                next_sat_date = pd.Timestamp(idx.date() + timedelta(days=days_until_sat))

                pos = self.holidayDatetime.searchsorted(idx, side='right')
                if pos < len(self.holidayDatetime):
                    next_holiday = self.holidayDatetime[pos]
                    next_breakday = next_sat_date if next_sat_date < next_holiday else next_holiday
                    next_breaktime = next_breakday- timedelta(hours=4)
                    post_break=(next_breaktime-idx).total_seconds() // 60


            week_fr,day_fr=calculate_week_fraction(idx)
            day_fr_sin = 0.5*(np.sin(day_fr * 2 * np.pi)+1)
            day_fr_cos = 0.5*(np.cos(day_fr * 2 * np.pi)+1)

            week_fr_sin = 0.5*(np.sin(week_fr * 2 * np.pi)+1)
            week_fr_cos = 0.5*(np.cos(week_fr * 2 * np.pi)+1)

            reference_time = pd.Timestamp('2000-01-01 00:00:00')
            absolute_time = (idx - reference_time).total_seconds() // 60

            
            self.close10=self.close10+row['close']-self.t3Base['close'].iat[-10]
            self.close60=self.close60+row['close']-self.t3Base['close'].iat[-60]
            self.close240=self.close240+row['close']-self.t3Base['close'].iat[-240]
            self.close1380=self.close1380+row['close']-self.t3Base['close'].iat[-1380]

            self.volume10=self.volume10+row['volume']-self.t3Base['volume'].iat[-10]
            self.volume60=self.volume60+row['volume']-self.t3Base['volume'].iat[-60]
            self.volume240=self.volume240+row['volume']-self.t3Base['volume'].iat[-240]
            self.volume1380=self.volume1380+row['volume']-self.t3Base['volume'].iat[-1380]

            self.volweek_raw=self.volweek_raw+log_ret-self.t3Base['log_ret'].iat[-self.lrt]
            self.close3d=self.close3d+row['close']-self.t3Base['close'].iat[-self.c3d]

            close_10=self.close10/10
            close_60=self.close60/60
            close_240=self.close240/240
            close_1380=self.close1380/1380
            volume_10=self.volume10/10
            volume_60=self.volume60/60
            volume_240=self.volume240/240
            volume_1380=self.volume1380/1380
            vlr=self.volweek_raw/self.lrt
            cld=self.close3d/self.c3d


            self.t3Base.append_row_keep_last(idx, [row['open'], row['high'], row['low'], row['close'], row['volume'],pre_event,post_event,pre_break,post_break,
                                                   day_fr_sin,day_fr_cos,week_fr_sin,week_fr_cos,absolute_time,
                                                   volume_10,volume_60,volume_240,volume_1380,
                                                   close_10,close_60,close_240,close_1380,
                                                   log_ret,vlr,cld])


        

    def sum_last(self, x: int, col: str) -> float:
        """
        返回 self.t2Base 中最后 x 行的 'close' 值的平均数。
        如果 x 大于行数，则使用全部行来计算。
        """
        return self.t3Base[col].iloc[-x:].sum()
        

if __name__ == '__main__':
    main_program=live_t3()

