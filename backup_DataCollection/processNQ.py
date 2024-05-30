
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil import tz
from datetime import datetime

def is_dst(us_date_str):
    """
    判断给定的美国日期（格式为"YYYYMMDD"）是否在美国的夏令时期间。
    
    参数:
    us_date_str (str): 美国日期字符串，格式为"YYYYMMDD"。
    
    返回:
    bool: 如果日期在夏令时期间返回True，否则返回False。
    """
    # 将输入的日期字符串转换为datetime对象
    date_obj = datetime.strptime(us_date_str, '%Y%m%d').replace(tzinfo=tz.gettz('US/Eastern'))  # 假设使用美国东部时间作为参考
    
    # 判断该日期是否在夏令时期间
    return date_obj.dst() != timedelta(0)

# 判断是否为美国夏令时期间（这里简化处理，具体需根据实际年份和 DST 规则判断）

# 时区转换函数
def timezone_adjustment(data_list):
    for row in data_list:
        date_str, time_str = row[0], row[1]
        if not is_dst(date_str):
            hour, minute = map(int, time_str.split(':'))
            hour += 1  # 非夏令时增加1小时
            time_str = f"{hour:02d}:{minute:02d}"
        yield [date_str, time_str] + row[2:]

# 数值补全函数
def interpolate_data(data_list):
    last_time = None
    for i, row in enumerate(data_list):
        date_str, time_str = row[:2]
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
        if last_time is not None and (dt - last_time).total_seconds() / 60 > 1:
            # 需要考虑是否在17:00至18:00之间，这里简化处理，直接补全
            while (dt - last_time).total_seconds() / 60 > 1:
                new_dt = last_time + timedelta(minutes=1)
                if '17:00' <= new_dt.time().strftime('%H:%M') <= '18:00':
                    break
                new_row = [new_dt.strftime('%Y%m%d'), new_dt.strftime('%H:%M')] + [0]*5 + [0]  # 第7列为0
                data_list.insert(i, new_row)
                last_time = new_dt
        else:
            last_time = dt
    return data_list

def process_file(input_file_path, output_folder):
    with open(input_file_path, 'r') as file:
        content = file.read()
    data_list = [line.strip().split(',') for line in content.splitlines()]
    
    # 时区转换
    data_list = list(timezone_adjustment(data_list))
    
    # 数值补全
    data_list = interpolate_data(data_list)
    
    output_file_path = os.path.join(output_folder, os.path.basename(input_file_path))
    with open(output_file_path, 'w') as file:
        for row in data_list:
            file.write(','.join(map(str, row)) + '\n')

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file in tqdm(files, desc="Processing files"):
        input_file_path = os.path.join(input_folder, file)
        process_file(input_file_path, output_folder)

if __name__ == "__main__":
    input_folder = 'Wang\\Individual\\NQ'  # 输入文件夹路径
    output_folder = 'data\\processNQ2'  # 输出文件夹路径    
    main(input_folder, output_folder)

# input_folder = 'Wang\\Individual\\NQ'  # 输入文件夹路径
# output_folder = 'data\\processNQ2'  # 输出文件夹路径
# process_and_save(input_folder, output_folder)


# def timezone_conversion(lines):
#     """时区转换：增加每行时间1小时"""
#     for i, line in enumerate(lines):
#         parts = line.strip().split(',')
#         date_str, time_str = parts[0], parts[1]
#         dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
#         dt += timedelta(hours=1)
#         parts[0] = dt.strftime("%Y%m%d")  # 日期部分保持原样，仅时间增加1小时
#         parts[1] = dt.strftime("%H%M")
#         lines[i] = ','.join(parts)
#     return lines

# def interpolate_data(lines):
#     """数值补全：按分钟递增补全不连续的时间，其他列数据复制上一行"""
#     if len(lines) <= 1:
#         return lines
    
#     result = [lines[0]]
#     prev_dt = datetime.strptime(lines[0].split(',')[0] + ' ' + lines[0].split(',')[1], "%Y%m%d %H%M")
    
#     for line in lines[1:]:
#         curr_dt = datetime.strptime(line.split(',')[0] + ' ' + line.split(',')[1], "%Y%m%d %H%M")
#         time_diff_min = int((curr_dt - prev_dt).total_seconds() / 60)
        
#         if 0 < time_diff_min < 30:  # 时间差小于半小时
#             # 按分钟递增补全时间，其他列数据复制上一行，最后一列赋值为0
#             while time_diff_min > 1:
#                 prev_dt += timedelta(minutes=1)
#                 interpolated_line = f"{prev_dt.strftime('%Y%m%d')}," \
#                                    f"{prev_dt.strftime('%H%M')}," 
#                                 #    f"{result[-1].split(',')[2]},"
#                 interpolated_line += ','.join(result[-1].split(',')[2:6]) + ",0"
#                 result.append(interpolated_line)
#                 time_diff_min -= 1
#         else:
#             result.append(line)
#         prev_dt = curr_dt
    
#     return result

# def process_and_save(input_path, output_path):
#     """处理文件并保存到指定文件夹"""
#     os.makedirs(output_path, exist_ok=True)
#     txt_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
    
#     for filename in tqdm.tqdm(txt_files, desc="Processing files", unit="file"):
#         with open(os.path.join(input_path, filename), 'r') as file:
#             lines = file.readlines()
        
#         lines = timezone_conversion(lines)
#         interpolated_lines = interpolate_data(lines)
#         # interpolated_lines = lines

        
#         with open(os.path.join(output_path, filename), 'w') as new_file:
#             new_file.writelines(line + '\n' for line in interpolated_lines)

# input_folder = 'Wang\\Individual\\NQ'  # 输入文件夹路径
# output_folder = 'data\\processNQ'  # 输出文件夹路径
# process_and_save(input_folder, output_folder)

