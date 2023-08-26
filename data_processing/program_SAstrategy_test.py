import numpy as np
import pandas as pd
import movingWindow as mw
import averageDecisionMachine as adm
from datetime import timedelta
import matplotlib.pyplot as plt
import backtest
df = pd.read_pickle('MNQtraindata')
#Set up the Container
window=mw.Window(df,1000)
#Set up the Strategy
strategy=adm.AverageDecisionMachine()
#Set up back tester
tester=backtest.BackTest(strategy,window)
result=tester.backTest(-0.003,0.03,-0.03)
#Print result
print('Total Profit = %.2f'%(result*100),"%",sep='')
