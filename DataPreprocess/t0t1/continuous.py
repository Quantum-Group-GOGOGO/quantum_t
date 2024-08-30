import pandas as pd
import datetime
import numpy as np
def is_inplaced(time):#用来设定特定的时间段不进行填充
    if time.weekday()==4:#Friday
        return False
    if time.weekday()==5:#Saturday
        return False
    if time.weekday()==6:#Sunday
        return False
    return True
    
def create_lines(df,start_time,end_time,time_interval,close_value):
    if end_time-start_time>=datetime.timedelta(minutes=30):#间断超过60分钟默认是休盘，不进行补齐处理
        return df
    df1=df
    current_time = start_time
    while current_time < end_time:
        new_row = pd.DataFrame([{
            'datetime': current_time,
            'open': close_value,
            'high': close_value,
            'low': close_value,
            'close': close_value,
            'volume': 0
        }])
        df1=pd.concat([df1, new_row], ignore_index=False)
        current_time += time_interval
    return df1

def continuous(data,time_interval):
    #建立一个新的dataframe用于拼接到原来的dataframe上
    columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(columns=columns)
    current_index=1
    while current_index < data.shape[0]:
        if data.at[current_index,'datetime']-data.at[current_index-1,'datetime']>time_interval:
            df=create_lines(df,data.at[current_index-1,'datetime'],data.at[current_index,'datetime'],time_interval,data.at[current_index-1,'close'])
        current_index += 1
    #把新建的df拼接到data的后面并且重新排序一次
    data = pd.concat([data,df], ignore_index=False)
    data = data.drop_duplicates(subset='datetime')
    data.sort_values(by='datetime', ascending=True,inplace=True)  
    data.reset_index(drop=True,  inplace=True)
    return data

#各种数据储存路径
target="NQ"#!!!!!!!!!!!!!!!!产品名改这里
data_name="NQ_1week_per_min.pkl"#!!!!!!!!!!!文件名改这里

data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
data_path=data_path_prefix+"/Type0/"+target+"/"
#数据名
data=pd.read_pickle(data_path+data_name)
data.reset_index(drop=True,  inplace=True)
print(data.head())

#时间间隔，这里点与点之间的间隔是一分钟
time_interval=datetime.timedelta(minutes=1)
print(data.head())
print(data.tail())
data=continuous(data,time_interval)
print(data.head())
print(data.tail())
data_path=data_path_prefix+"/Type1/"+target+"/"
data.to_pickle(data_path+data_name)