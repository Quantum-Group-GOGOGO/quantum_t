from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
from tqdm import tqdm

from utils import HistoricalDataCollector

def NQ_CQG_read(file_path):
    df=pd.read_csv(file_path,header=None, names=['date','time','open','high','low','close','volume'],dtype={'date':object,'time':object})
    df['datetime']=df[['date','time']].apply(lambda x:datetime.datetime.strptime(x['date']+x['time'],'%Y%m%d%H%M'),axis=1)
    df=df[['datetime','open','high','low','close','volume']]
    return df

def NQ_CQG_filename_str(year,season):
    str1=str(year)
    match season:
        case 0:
            return 'NQ'+str1+'H.txt'
        case 1:
            return 'NQ'+str1+'M.txt'
        case 2:
            return 'NQ'+str1+'U.txt'
        case 3:
            return 'NQ'+str1+'Z.txt'
        case _:
            return 'NQ'+str1+'.txt'
        
def NQ_future_ystoi(year,season):
    begin_year=1999
    begin_season=2
    #0,1,2,3 for the future end in March, June, September and December
    return (year-begin_year)*4+(season-begin_season)

def NQ_future_itoys(index):
    begin_year=1999
    begin_season=2
    #0,1,2,3 for the future end in March, June, September and December
    return begin_year+((index+begin_season)//4), (begin_season+index)%4

def NQ_future_select(date):
    year=date.year
    month=date.month
    day=date.day
    if month == 3 or month == 6 or month == 9 or month == 12:
        if day >= 15:
            month=month+1
    if month ==13:
        month=1
        year=year+1
    season=month//3
    return year,season

def NQ_CQG_read_all():
    newest=NQ_future_ystoi(2024,1)
    data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
    data_path=data_path_prefix+"/raw/NQ_historic/Individual/NQ/"
    df_set={}
    for i in tqdm(range(newest+1)):
        y,s = NQ_future_itoys(i)
        file=data_path+NQ_CQG_filename_str(y,s)
        df_set[i]=NQ_CQG_read(file)
    return df_set

def NQ_CQG_read_fromDate(end_date,num_days):#
    start_date=end_date-timedelta(days=num_days)
    y,s=NQ_future_select(end_date)
    end_index=NQ_future_ystoi(y,s)
    y,s=NQ_future_select(start_date)
    start_index=NQ_future_ystoi(y,s)

    data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
    data_path=data_path_prefix+"/raw/NQ_historic/Individual/NQ/"
    df_set={}

    for i in tqdm(range(start_index,end_index+1)):
        y,s = NQ_future_itoys(i)
        file=data_path+NQ_CQG_filename_str(y,s)
        df_set[i]=NQ_CQG_read(file)
    return df_set

def pick_1day_from_NQset(df_set,date):
    y,s=NQ_future_select(date)
    index=NQ_future_ystoi(y,s)
    return df_set[index].loc[df_set[index].datetime.apply(lambda x: True if x.date()==date.date() else False)]

def readNQ_xday_from_CQG(end_date_str,num_days):
    end_date=datetime.datetime.strptime(end_date_str,'%Y%m%d')
    print("Read the data from file")
    df_set=NQ_CQG_read_fromDate(end_date,num_days)
    dfs=[]
    print("Arrange the data")
    for i in tqdm(range(num_days)):
        index_date = end_date - timedelta(days=i)
        df=pick_1day_from_NQset(df_set,index_date)
        if df is not None:
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='datetime', keep='first')
    combined_df.sort_values(by='datetime', inplace=True, ascending=False)
    return combined_df

#data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
#data_path=data_path_prefix+"/raw/NQ_historic/Individual/NQ/NQ1999U.txt"
#df1=NQ_CQG_read(data_path)
#df={1:df1,2:df2}
#df_set=NQ_CQG_read_all()
#print(df_set[1].head(5))

#print(NQ_CQG_filename_str(1998,2))
#dt=datetime.datetime.strptime('19991022','%Y%m%d')
#print(dt.date())
#print(NQ_CQG_read_fromDate(dt,0))
df=readNQ_xday_from_CQG('20240501',130)
print(df.head())
df.to_pickle('./tempdata/NQ2024.pkl')
#df=readNQ_xday_from_CQG('20000101',180)
#print(df.head())
#df.to_pickle('./tempdata/NQ1999.pkl')
#for i in range(2024,2025):
#    st1=str(i)+'0101'
#    df=readNQ_xday_from_CQG(st1,366)
#    st2='./tempdata/NQ'+str(i-1)+'.pkl'
#    df.to_pickle(st2)
