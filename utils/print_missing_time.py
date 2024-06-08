import pandas as pd
from datetime import datetime, timedelta, time
def print_missing_time(year):
    df=pd.read_pickle("./tempdata/NQ"+str(year)+".pkl")

    start_time = df['datetime'].dt.floor('D').min()
    end_time = df['datetime'].dt.ceil('D').max()
    all_times = pd.date_range(start=start_time, end=end_time, freq='15min')

    # 找出哪些时间段不在数据框中存在记录
    existing_times = df['datetime'].dt.floor('15min').unique()
    missing_times = [time for time in all_times if time not in existing_times]

    DateF= pd.DataFrame(columns=['datetime','weekday'])
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for now_time in missing_times:
        if now_time.weekday() < 4:
            datetime_obj = now_time.to_pydatetime()

            DateF = pd.concat([DateF, pd.DataFrame({'datetime': [datetime_obj],'weekday': [weekdays[now_time.weekday()]]})], ignore_index=True)
        elif now_time.weekday()==4:
            if now_time.time()<time(20, 0):
                datetime_obj = now_time.to_pydatetime()
                DateF = pd.concat([DateF, pd.DataFrame({'datetime': [datetime_obj],'weekday': ['Friday']})], ignore_index=True)
        elif now_time.weekday()==6:
            if now_time.time()>time(15, 0):
                datetime_obj = now_time.to_pydatetime()
                DateF = pd.concat([DateF, pd.DataFrame({'datetime': [datetime_obj],'weekday': ['Sunday']})], ignore_index=True)
    print(DateF.head())      
    DateF.to_csv('./datacheck/NQ'+str(year)+'TimeCheck.csv')
    #missing_times = [time for time in all_times if time not in existing_times]

    # 或者将缺失的时间段存储到一个数据框中
    #missing_times_df = pd.DataFrame(missing_times, columns=['Missing Times'])
    #print(missing_times_df)
for i in range(1999,2025):
    print_missing_time(i)