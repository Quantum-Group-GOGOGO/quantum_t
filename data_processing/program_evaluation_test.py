import numpy as np
import pandas as pd
import evaluationDoubleWindow as mw
import datetime
from datetime import timedelta

df = pd.read_pickle('MNQtraindata')
#Set up the Container
window=mw.DoubleWindow(df,500,500)
window.init_evaluation(500,0.002)

result=np.empty([10000,3])
result=np.append(result,[[1,2,3]],axis=0)

index=0
while not window.is_end():
    print(index)
    result[index,0]=index
    result[index,1]=window.average(0)
    result[index,2]=window.value()
    window.march()
    index+=1
result=result[0:index,:]
np.savetxt("evaluation.csv",result,delimiter=',')


