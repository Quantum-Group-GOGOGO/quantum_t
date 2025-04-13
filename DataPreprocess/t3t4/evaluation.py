import printh as ph
import pandas as pd
import numpy as np
# 数据读取
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
#T3_data_path=data_base+'/type3/Nasdaq_qqq_align_labeled_base.pkl'
T4_data_path = data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated_history.pkl'
df = pd.read_pickle(T4_data_path)
printH=ph.PrintH(df)
printH.add_hidden_column('close')
printH.add_hidden_column('time_fraction')
printH.add_hidden_column('post_event')
printH.add_hidden_column('post_break')
printH.add_hidden_column('open')
printH.add_hidden_column('volume')
printH.add_hidden_column('high')
printH.add_hidden_column('low')
printH.add_hidden_column('pre_event')
printH.add_hidden_column('event')
printH.add_hidden_column('pre_break')
printH.add_hidden_column('time_break_flag')
printH.add_hidden_column('sinT')
printH.add_hidden_column('cosT')

printH.print()

