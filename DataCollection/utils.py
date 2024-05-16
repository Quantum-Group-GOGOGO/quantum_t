import pandas as pd
import datetime
from datetime import timedelta
import pandas as pd
# from DatacollectionQQQ1D import DatacollectionQQQ1Day as QQQ1D
from ib_insync import *
def DatacollectionQQQ1Day(IBobject, date, barSize):
    contract = Contract()
    contract  = Stock('QQQ','SMART','USD')
    bars = IBobject.reqHistoricalData(
    contract, endDateTime=(date+' 00:00:00'), durationStr='1 D',
    barSizeSetting=barSize, whatToShow='TRADES', useRTH=False)

    df = util.df(bars)
    return df

def Concat_DF_Sort(df1,df2):
# Concatenate the DataFrames
    df_combined = pd.concat([df1, df2])

    # Sort by date
    df_combined = df_combined.sort_values(by='date')
    df_combined = df_combined.drop_duplicates(subset='date', keep='first')
    return df_combined

def QQQXD(IBobject,initial_date,date_num, barSize):
    for date_num_index in range(date_num):
        date_index=datesub(initial_date,date_num_index)
        if date_num_index==0:
            df=DatacollectionQQQ1Day(IBobject,date_index, barSize)
        else:
            df=Concat_DF_Sort(df,DatacollectionQQQ1Day(IBobject,date_index, barSize))
    df=df.reset_index().drop('index', axis=1)
    return df

def dateadd(initial_date,date_add):
    start_date = datetime.datetime.strptime(initial_date, '%Y%m%d').date()
    delta = datetime.timedelta(days=date_add)
    new_date = start_date + delta
    return new_date.strftime('%Y%m%d')
def datesub(initial_date,date_sub):
    start_date = datetime.datetime.strptime(initial_date, '%Y%m%d').date()
    delta = datetime.timedelta(days=date_sub)
    new_date = start_date - delta
    return new_date.strftime('%Y%m%d')
def is_weekend(date_str):
    date_format = '%Y%m%d'
    date = datetime.datetime.strptime(date_str, date_format).date()
    # Check if the day of the week is Saturday (5) or Sunday (6)
    return date.weekday() == 5 or date.weekday() == 6