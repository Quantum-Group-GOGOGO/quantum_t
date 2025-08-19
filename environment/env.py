import os
import sys

os_name = os.getenv('PYTHON_HARDWARE', '')

if os_name == 'MACOS_Laptop':
    print('Configuration to Macbook Laptop')
    data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    live_data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_live_data'
    project_base = '/Users/wentianwang/Soft/Quantum/quantum_t'
    localhost='127.0.0.1'
elif os_name == 'Windows_Desktop':
    print('Configuration to Home Desktop Windows')
    data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
    live_data_base = 'D:/quantum/quantum_t_data/quantum_t_live_data'
    project_base = 'D:/quantum/quantum_t/'
    localhost='127.0.0.1'
elif os_name == 'HOME_WINDOWS_WSL':
    print('Configuration to Home Desktop WSL')
    data_base = '/mnt/d/quantum/quantum_t_data/quantum_t_data'
    live_data_base = '/mnt/d/quantum/quantum_t_data/quantum_t_live_data'
    project_base = '/mnt/d/quantum/quantum_t/'
    localhost=os.getenv('local_IP', '')
    #localhost='127.0.0.1'
elif os_name == 'Ubuntu_Desktop':
    print('Configuration to Home Desktop Ubuntu')
    data_base = '/mnt/gdrive/quantum_t_data'
    live_data_base = '/mnt/gdrive/quantum_t_live_data'
    project_base = '/mnt/gdrive/'
    localhost='127.0.0.1'
else:
    raise ValueError("Unknown Environment, Please setup the environment variable: PYTHON_HARDWARE")

sys.path.insert(0, project_base + '/dataloaders')
sys.path.insert(0, project_base + '/models')