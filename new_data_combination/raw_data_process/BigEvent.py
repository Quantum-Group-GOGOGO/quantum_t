import re
from datetime import datetime, date, time as dtime
import pandas as pd
from env import *

# 英文星期列表，用于检测“日期行”
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# 时间行的正则：以 HH:MM 开头
TIME_LINE_RE = re.compile(r'^\d{1,2}:\d{2}')
df = pd.DataFrame(
    index=pd.DatetimeIndex([], name='datetime'),
    columns=['item'],
    dtype=int
)
df2 = pd.DataFrame(
    index=pd.DatetimeIndex([], name='datetime'),
    columns=['item'],
    dtype=int
)
def process_file(filename: str):
    global live_data_base
    current_date: date | None = None

    with open(filename, 'r', encoding='utf-8') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip('\n')

            # —— 1. 日期行：以 Monday…Sunday 开头，更新 current_date
            if re.match(r'^(?:' + '|'.join(WEEKDAYS) + r')\b', line):
                parts = line.split(',', 2)
                if len(parts) == 3:
                    month_day = parts[1].strip()   # e.g. "December 9"
                    year_str  = parts[2].strip()   # e.g. "2024"
                    date_str  = f"{month_day} {year_str}"
                    try:
                        current_date = datetime.strptime(date_str, "%B %d %Y").date()
                        #print(f"[Line {lineno}] 更新日期：{current_date}")
                    except ValueError as e:
                        print(f"[Line {lineno}] 日期解析失败 “{date_str}”：{e}")
                        current_date = None
                else:
                    print(f"[Line {lineno}] 日期行逗号不足两处，无法更新日期。")
                    current_date = None

            # —— 2. 时间行：以 HH:MM 开头，用 current_date 组合 datetime
            elif TIME_LINE_RE.match(line):
                # 如果还没读到有效的日期，就跳过
                if current_date is None:
                    print(f"[Line {lineno}] 尚未设置日期，跳过时间行解析。")
                    continue

                # 按制表符拆分
                fields = re.split(r'\t+', line)
                if len(fields) >= 3:
                    time_str  = fields[0].strip()   # e.g. "08:30"
                    item_name = fields[2].strip()   # e.g. "Initial Jobless Claims"

                    # 解析 time_str 为 time 对象
                    try:
                        t = datetime.strptime(time_str, "%H:%M").time()
                        combined_dt = datetime.combine(current_date, t)
                        if 'Fed Interest Rate Decision' in item_name:
                            #print(f"[Line {lineno}] 事件时间：{combined_dt!r}  |  项目：{item_name}")
                            #df.index = df.index.insert(len(df.index), combined_dt)
                            df.loc[combined_dt, 'item'] = 1
                        elif 'Core CPI' in item_name:
                            #print(f"[Line {lineno}] 事件时间：{combined_dt!r}  |  项目：{item_name}")
                            #df.index = df.index.insert(len(df.index), combined_dt)
                            df.loc[combined_dt, 'item'] = 2
                        elif 'Nonfarm Payrolls' in item_name:
                            #print(f"[Line {lineno}] 事件时间：{combined_dt!r}  |  项目：{item_name}")
                            #df.index = df.index.insert(len(df.index), combined_dt)
                            df.loc[combined_dt, 'item'] = 3    
                    except ValueError as e:
                        print(f"[Line {lineno}] 无法解析时间 “{time_str}”：{e}")
                else:
                    print(f"[Line {lineno}] 时间行字段不足，跳过。")

            elif line.startswith('All Day'):
                # 用制表符切分，第二段应包含 'Holiday'
                parts = re.split(r'\t+', line)
                #print(f"{current_date} {parts[2]}")
                if current_date is not None and len(parts) > 1 and 'Holiday' in parts[2]:
                    #print(f"[Line {lineno}] Holiday on {current_date}")
                    df2.loc[current_date, 'item'] = 4 

            # —— 3. 其他行：忽略
            else:
                continue
    event_file_path=live_data_base+'/big_event/events.pkl'
    holiday_file_path=live_data_base+'/big_event/holiday.pkl'
    df.to_pickle(event_file_path)
    df2.to_pickle(holiday_file_path)
if __name__ == "__main__":
    process_file('date.txt')