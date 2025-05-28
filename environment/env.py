import os
import sys

os_name = os.getenv('PYTHON_HARDWARE', '')

if os_name == 'MACOS_Laptop':
    print('A')
    data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    project_base = '/Users/wentianwang/Soft/Quantum/quantum_t'
    sys.path.insert(0, project_base + '/training/dataloaders')
    sys.path.insert(0, project_base + '/training/models')
elif os_name == 'Windows_Desktop':
    print('B')
    data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
else:
    raise ValueError("Unknown Environment, Please setup the environment variable: PYTHON_HARDWARE")