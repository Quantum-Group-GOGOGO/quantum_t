import datetime
from datetime import timedelta
import pandas as pd
from DatacollectionQQQ1D import DatacollectionQQQ1Day as QQQ1D
from ConcatDF import Concat_DF
from date_calculate_string import *


def QQQXD(IBobject,initial_date,date_num, barSize):
    for date_num_index in range(date_num):
        date_index=datesub(initial_date,date_num_index)
        if date_num_index==0:
            df=QQQ1D.DatacollectionQQQ1Day(IBobject,date_index, barSize)
        else:
            df=Concat_DF.Concat_DF_Sort(df,QQQ1D.DatacollectionQQQ1Day(IBobject,date_index, barSize))
    df=df.reset_index().drop('index', axis=1)
    return df