import pandas as pd
from datetime import datetime, timedelta
df=pd.read_pickle("NQ2024.pkl")

start_time = df['datetime'].dt.floor('D').min()
end_time = df['datetime'].dt.ceil('D').max()
all_times = pd.date_range(start=start_time, end=end_time, freq='15min')

# 找出哪些时间段不在数据框中存在记录
existing_times = df['datetime'].dt.floor('15min').unique()
missing_times = [time for time in all_times if time not in existing_times]

list= pd.DataFrame(columns=['datetime'])

for time in missing_times:
    if time.weekday() < 4:
        datetime_obj = time.to_pydatetime()
        list._append({'datetime': datetime_obj},ignore_index=True)
        
print(list.head())       
#missing_times = [time for time in all_times if time not in existing_times]

# 或者将缺失的时间段存储到一个数据框中
#missing_times_df = pd.DataFrame(missing_times, columns=['Missing Times'])
#print(missing_times_df)