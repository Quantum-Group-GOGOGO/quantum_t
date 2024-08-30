import os
from datetime import datetime, timedelta
from tqdm import tqdm

def add_one_hour_and_save(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in tqdm(os.listdir(input_folder), desc="Processing files"):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            with open(input_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            processed_lines = []
            for line in lines:
                parts = line.strip().split(',')
                date_str, time_str = parts[0], parts[1]
                
                # 将日期字符串转换为datetime对象，增加1小时，再转回字符串
                dt = datetime.strptime(f'{date_str} {time_str}', '%Y%m%d %H%M')
                dt += timedelta(hours=1)
                
                # 更新日期和时间
                new_date_str = dt.strftime('%Y%m%d')
                new_time_str = dt.strftime('%H%M')
                
                # 重新组合行数据
                new_line = f'{new_date_str},{new_time_str},{",".join(parts[2:])}\n'
                processed_lines.append(new_line)
            
            # 写入新文件
            with open(output_path, 'w', encoding='utf-8') as file:
                file.writelines(processed_lines)

# 使用示例
#input_folder = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data/raw/NQ_historic/Individual/NQ/'  # 输入文件夹路径
#output_folder = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data/raw/NQ_historic/Individual/NQ_EDT/'  # 输出文件夹路径
add_one_hour_and_save(input_folder, output_folder)