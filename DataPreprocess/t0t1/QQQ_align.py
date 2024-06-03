import pandas as pd
import datetime

data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data"
data_path=data_path_prefix+"/Type0/QQQ/"
QQQ=pd.read_pickle(data_path+"QQQ_1week_per_min.pkl")
data_path=data_path_prefix+"/Type0/NDX/"
NDX=pd.read_pickle(data_path+"NDX_1week_per_min.pkl")

QQQ['date']=QQQ['date']-datetime.timedelta(hours=12)
NDX['date']=NDX['date']-datetime.timedelta(hours=12)
QQQ=QQQ.reindex(reversed(QQQ.index))
NDX=NDX.reindex(reversed(NDX.index))
print(QQQ.head())
print(NDX.head())

def is_weekend(date_str):
        date_format = '%Y%m%d'
        date = datetime.strptime(date_str, date_format).date()
        return date.weekday() == 5 or date.weekday() == 6

def QQQ_Align(column_name,NDX,QQQ,timepoint):
    time_destination = datetime.datetime.strptime(timepoint, '%H%M%S').time()
    date_start=QQQ['date'].iloc[0].date()
    date_end=QQQ['date'].iloc[-1].date()
    if(date_start != NDX['date'].iloc[0].date()):
         print('ERROR: QQQ and NDX start from differnt date')
         return
    if(date_end != NDX['date'].iloc[-1].date()):
         print('ERROR: QQQ and NDX end from differnt date')
         return
    day_num=(date_end-date_start).days
    QQQ_index=0
    NDX_index=0
    
    adjust_index=0
    list= pd.DataFrame(columns=['date','K'])
    for day_i in range(day_num+1):
        date_i=date_start+datetime.timedelta(days=day_i)
        QQQ_found=0
        NDX_found=0
        while(QQQ['date'].iloc[QQQ_index].date()<date_i):
            QQQ_index=QQQ_index+1
        while(NDX['date'].iloc[NDX_index].date()<date_i):
            NDX_index=NDX_index+1
        QQQ_start=QQQ_index
        NDX_start=NDX_index
        if date_i.weekday() == 5 or date_i.weekday() == 6:
            continue
        while(QQQ_found==0 and QQQ_index<QQQ.shape[0] and QQQ['date'].iloc[QQQ_index].date()==date_i):
            if(QQQ['date'].iloc[QQQ_index].time()>=time_destination):
                QQQ_found=1
                QQQ_index=QQQ_index-1
            QQQ_index=QQQ_index+1
        while(NDX_found == 0 and NDX_index<NDX.shape[0] and NDX['date'].iloc[NDX_index].date()==date_i):
            if(NDX['date'].iloc[NDX_index].time()>=time_destination):
                NDX_found=1
                NDX_index=NDX_index-1
            NDX_index=NDX_index+1
        if(QQQ_found==0 or NDX_found==0):
            print('ERROR: QQQ or NDX cannot find a time to align',date_i)
            return
        else:
            print('Adjust for date: ',date_i)
        
        while(QQQ['date'].iloc[QQQ_index] != NDX['date'].iloc[NDX_index]):
            Print('Waring!')
            if(QQQ['date'].iloc[QQQ_index] > NDX['date'].iloc[NDX_index]):
                NDX_found=0
                time_destination=QQQ['date'].iloc[QQQ_index].time()
                while(QQQ['date'].iloc[QQQ_index] < NDX['date'].iloc[NDX_index] and NDX_found==0 and NDX_index<NDX.shape[0] and NDX['date'].iloc[NDX_index].date()==date_i):
                    NDX_index=NDX_index+1
                    if(NDX['date'].iloc[NDX_index].time() >= time_destination):
                        NDX_found=1
                        break
            else:
                QQQ_found=0
                time_destination=NDX['date'].iloc[NDX_index].time()
                while(NDX['date'].iloc[NDX_index] < QQQ['date'].iloc[QQQ_index] and QQQ_found==0 and QQQ_index<QQQ.shape[0] and QQQ['date'].iloc[QQQ_index].date()==date_i):
                    QQQ_index=QQQ_index+1
                    if(QQQ['date'].iloc[QQQ_index].time() >= time_destination):
                        QQQ_found=1
                        break
            if(QQQ_found==0 or NDX_found==0):
                print('ERROR: QQQ or NDX cannot find a time to align after the time adjusted')
                return
        k=NDX[column_name].iloc[NDX_index]/QQQ[column_name].iloc[QQQ_index]
        print('K=', k)
        newdata=pd.DataFrame({'date': [day_i],'K':[k],'time1': [NDX['date'].iloc[NDX_index]],'time2': [QQQ['date'].iloc[QQQ_index]],'QQQ_start': [int(QQQ_start)],'NDX_start': [int(NDX_start)]})
        list=pd.concat([list, newdata],ignore_index=True)
    print(list.head())
    #这一行之前都是用来制作list信息的，用以存放每一天的对齐时间和QQQ的行数范围（从哪一行开始算这一天的）
    #通过制定的list的信息来进行缩放
    for day_i in range(list.shape[0]):
        if day_i != list.shape[0]-1:
               #QQQ.loc[int(list['QQQ_start'].iloc[day_i]):int(list['QQQ_start'].iloc[day_i+1])-1, column_name]*=list['K'].iloc[day_i]
                QQQ.iloc[int(list['QQQ_start'].iloc[day_i]):int(list['QQQ_start'].iloc[day_i+1]), QQQ.columns.get_loc(column_name)] *= list['K'].iloc[day_i]
        else:
                QQQ.iloc[int(list['QQQ_start'].iloc[day_i]):QQQ.shape[0], QQQ.columns.get_loc(column_name)] *= list['K'].iloc[day_i]

#print(QQQ.iloc[958:962])
print(QQQ.head())
print(QQQ.tail())