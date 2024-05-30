import os
from tqdm import tqdm
from datetime import datetime, timedelta

def parse_time(time_str):
    """解析时间字符串为datetime对象"""
    return datetime.strptime(time_str, '%H%M')

def fill_gaps(input_folder, output_folder):
    """处理文件夹中的txt文件，按要求填充空缺行并输出到新文件夹"""
    for filename in tqdm(os.listdir(input_folder), desc="Processing files"):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 读取并处理文件
            with open(input_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 初始化变量
            prev_date_str, prev_time_str, prev_data = None, None, None
            prev_time = None  # 初始化prev_time变量
            new_lines = []
            
            for line in lines:
                parts = line.strip().split(',')
                date_str, time_str, *data_parts = parts
                curr_time = parse_time(time_str)
                
                # 处理时间差
                if prev_time is None or (curr_time - prev_time).total_seconds() / 60 > 30:
                    # 直接添加当前行，并更新prev变量
                    new_lines.append(line)
                    prev_date_str, prev_time_str = date_str, time_str
                    prev_time = curr_time  # 更新prev_time
                    prev_data = data_parts
                else:
                    # 填充缺失的行
                    for minute_offset in range(1, int((curr_time - prev_time).total_seconds() / 60)):
                        fill_time = prev_time + timedelta(minutes=minute_offset)
                        fill_time_str = fill_time.strftime('%H%M')
                        new_lines.append(f"{date_str},{fill_time_str},{','.join(prev_data)[:-2]},0\n")
                    # 添加当前行
                    new_lines.append(line)
                    prev_date_str, prev_time_str = date_str, time_str
                    prev_time = curr_time  # 更新prev_time
                    prev_data = data_parts
            
            # 写入新文件
            with open(output_path, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)

# 调用函数时，请确保输入和输出文件夹路径正确无误


# 使用示例
input_folder = 'data\\add_1_hour'
output_folder = 'data\\interpolate_missing_times'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

fill_gaps(input_folder, output_folder)
