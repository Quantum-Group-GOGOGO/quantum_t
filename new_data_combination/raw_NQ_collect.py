from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from config import parse_args
from utils import HistoricalDataCollector
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'


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
    start_date=end_date-timedelta(days=num_days+1)
    y,s=NQ_future_select(end_date)
    end_index=NQ_future_ystoi(y,s)
    y,s=NQ_future_select(start_date)
    start_index=NQ_future_ystoi(y,s)

    data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
    data_path=data_path_prefix+"/raw/NQ_historic/Individual/NQ_EDT/"
    df_set={}

    for i in tqdm(range(start_index,end_index+1)):
        y,s = NQ_future_itoys(i)
        file=data_path+NQ_CQG_filename_str(y,s)
        df_set[i]=NQ_CQG_read(file)
    return df_set

def pick_1day_from_NQset(df_set, date):
    y, s = NQ_future_select(date)
    index = NQ_future_ystoi(y, s)
    
    # Define the start and end times
    start_datetime = datetime.datetime(date.year, date.month, date.day, 19) - timedelta(days=1)  # 前一天的19:00
    end_datetime = datetime.datetime(date.year, date.month, date.day, 18, 59)  # 当天的18:59
    
    # Select the relevant data from the DataFrame
    df_selected = df_set[index].loc[df_set[index]['datetime'].between(start_datetime, end_datetime)]
    
    return df_selected

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
    combined_df.sort_values(by='datetime', inplace=True, ascending=True)
    return combined_df
#————————————————————————————————The above are the methods to collect the data from CQG's database——————————————————————
#————————————————————————————————The below are the methods to collect the data from Our database————————————————————————
def NQ_DB_read_fromDate(end_date,num_days):
    start_date=end_date-timedelta(days=num_days+1)
    y,s=NQ_future_select(end_date)
    end_index=NQ_future_ystoi(y,s)
    y,s=NQ_future_select(start_date)
    start_index=NQ_future_ystoi(y,s)

    data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
    data_path=data_path_prefix+"/raw/NQ_historic/Individual/NQ_EDT_IBDB/"
    df_set={}

    for i in tqdm(range(start_index,end_index+1)):
        y,s = NQ_future_itoys(i)
        file=data_path+NQ_CQG_filename_str(y,s)
        df_set[i]=NQ_CQG_read(file)
    return df_set

def readNQ_xday_from_db(end_date_str,num_days):
    end_date=datetime.datetime.strptime(end_date_str,'%Y%m%d')
    df_set=NQ_DB_read_fromDate(end_date,num_days)
    dfs=[]
    print("Arrange the data")
    for i in tqdm(range(num_days)):
        index_date = end_date - timedelta(days=i)
        df=pick_1day_from_NQset(df_set,index_date)
        if df is not None:
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='datetime', keep='first')
    combined_df.sort_values(by='datetime', inplace=True, ascending=True)
    return combined_df

    data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
    data_path=data_path_prefix+"/raw/NQ_historic/Individual/NQ_EDT/"

#————————————————————————————————The above are the methods to collect the data from Our database——————————————————————
#————————————————————————————————The below are the methods to collect the data from IB's database——————————————————————
df=readNQ_xday_from_CQG('20240430',8780)
df.to_pickle(data_base+'/Type0/NQ/NQ_BASE.pkl')