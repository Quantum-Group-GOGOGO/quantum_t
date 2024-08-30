import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

data_path_prefix="/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data/Type1/"
data_path=data_path_prefix+"NQ/"
NQ=pd.read_pickle(data_path+"NQ_1week_per_min.pkl")
data_path=data_path_prefix+"NDX/"
NDX=pd.read_pickle(data_path+"NDX_1week_per_min.pkl")
data_path=data_path_prefix+"QQQ/"
QQQ=pd.read_pickle(data_path+"QQQ_1week_per_min.pkl")
#NQ['date']=NQ['date']-datetime.timedelta(hours=12)
#NDX['date']=NDX['date']-datetime.timedelta(hours=12)
#NQ=NQ.reindex(reversed(NQ.index))
#NDX=NDX.reindex(reversed(NDX.index))
print(NQ.tail())
print(QQQ.head())

def is_weekend(date_str):
        date_format = '%Y%m%d'
        date = datetime.strptime(date_str, date_format).date()
        return date.weekday() == 5 or date.weekday() == 6

def NQ_Align_V(column_name,NDX,NQ,timepoint1,timepoint2):
    time_destination1 = datetime.datetime.strptime(timepoint1, '%H%M%S').time()
    time_destination2 = datetime.datetime.strptime(timepoint2, '%H%M%S').time()
    date_start=(NQ['datetime'].iloc[0]+datetime.timedelta(hours=6)).date()
    date_end=NQ['datetime'].iloc[-1].date()
    if(date_start != NDX['datetime'].iloc[0].date()):
         print('ERROR: NQ and NDX start from differnt date')
         return
    if(date_end != NDX['datetime'].iloc[-1].date()):
         print('ERROR: NQ and NDX end from differnt date')
         return
    day_num=(date_end-date_start).days
    NQ_index=0
    NDX_index=0
    
    adjust_index=0
    list1= pd.DataFrame(columns=['datetime'])
    list2= pd.DataFrame(columns=['datetime'])
    for day_i in range(day_num+1):
        date_i=date_start+datetime.timedelta(days=day_i)
        NQ_found=0
        NDX_found=0
        while((NQ['datetime'].iloc[NQ_index]+datetime.timedelta(hours=6)).date()<date_i):
            NQ_index=NQ_index+1
        while(NDX['datetime'].iloc[NDX_index].date()<date_i):
            NDX_index=NDX_index+1
        NQ_start=NQ_index
        NDX_start=NDX_index
        if date_i.weekday() == 5 or date_i.weekday() == 6:
            continue
        while(NQ_found==0 and NQ_index<NQ.shape[0] and (NQ['datetime'].iloc[NQ_index]+datetime.timedelta(hours=6)).date()==date_i):
            if(NQ['datetime'].iloc[NQ_index]>=datetime.datetime.combine(date_i, time_destination1)):
                NQ_found=1
                NQ_index=NQ_index-1
            NQ_index=NQ_index+1
        while(NDX_found == 0 and NDX_index<NDX.shape[0] and NDX['datetime'].iloc[NDX_index].date()==date_i):
            if(NDX['datetime'].iloc[NDX_index]>=datetime.datetime.combine(date_i, time_destination1)):
                NDX_found=1
                NDX_index=NDX_index-1
            NDX_index=NDX_index+1
        if(NQ_found==0 or NDX_found==0):
            print('ERROR: NQ or NDX cannot find a time to align',date_i)
            return
        else:
            print('Adjust for date: ',date_i)
        while(NQ['datetime'].iloc[NQ_index] != NDX['datetime'].iloc[NDX_index]):
            print('Waring!')
            if(NQ['datetime'].iloc[NQ_index] > NDX['datetime'].iloc[NDX_index]):
                NDX_found=0
                time_destination1=NQ['date'].iloc[NQ_index].time()
                while(NQ['datetime'].iloc[NQ_index] < NDX['datetime'].iloc[NDX_index] and NDX_found==0 and NDX_index<NDX.shape[0] and NDX['date'].iloc[NDX_index].date()==date_i):
                    NDX_index=NDX_index+1
                    if(NDX['datetime'].iloc[NDX_index].time() >= time_destination1):
                        NDX_found=1
                        break
            else:
                NQ_found=0
                time_destination1=NDX['datetime'].iloc[NDX_index].time()
                while(NDX['datetime'].iloc[NDX_index] < NQ['datetime'].iloc[NQ_index] and NQ_found==0 and NQ_index<NQ.shape[0] and NQ['date'].iloc[NQ_index].date()==date_i):
                    NQ_index=NQ_index+1
                    if(NQ['datetime'].iloc[NQ_index].time() >= time_destination1):
                        NQ_found=1
                        break
            if(NQ_found==0 or NDX_found==0):
                print('ERROR: NQ or NDX cannot find a time to align after the time adjusted')
                return
        NDX_index1=NDX_index
        NQ_index1=NQ_index
        #k=NDX[column_name].iloc[NDX_index]/NQ[column_name].iloc[NQ_index]
        #k=NDX['close'].iloc[NDX_index]/NQ['close'].iloc[NQ_index]
        #print('K=', k)
        newdata=pd.DataFrame({'datetime': [day_i], 'NDX_index1': [NDX_index1], 'NQ_index1': [NQ_index1], 'time1': [NDX['datetime'].iloc[NDX_index]],'time2': [NQ['datetime'].iloc[NQ_index]],'NQ_start': [int(NQ_start)],'NDX_start': [int(NDX_start)]})
        list1=pd.concat([list1, newdata],ignore_index=True)
    NQ_index=0
    NDX_index=0
    for day_i in range(day_num+1):
        date_i=date_start+datetime.timedelta(days=day_i)
        NQ_found=0
        NDX_found=0
        while((NQ['datetime'].iloc[NQ_index]+datetime.timedelta(hours=6)).date()<date_i):
            NQ_index=NQ_index+1
        while(NDX['datetime'].iloc[NDX_index].date()<date_i):
            NDX_index=NDX_index+1
        NQ_start=NQ_index
        NDX_start=NDX_index
        if date_i.weekday() == 5 or date_i.weekday() == 6:
            continue
        while(NQ_found==0 and NQ_index<NQ.shape[0] and (NQ['datetime'].iloc[NQ_index]+datetime.timedelta(hours=6)).date()==date_i):
            if(NQ['datetime'].iloc[NQ_index]>=datetime.datetime.combine(date_i, time_destination2)):
                NQ_found=1
                NQ_index=NQ_index-1
            NQ_index=NQ_index+1
        while(NDX_found == 0 and NDX_index<NDX.shape[0] and NDX['datetime'].iloc[NDX_index].date()==date_i):
            if(NDX['datetime'].iloc[NDX_index]>=datetime.datetime.combine(date_i, time_destination2)):
                NDX_found=1
                NDX_index=NDX_index-1
            NDX_index=NDX_index+1
        if(NQ_found==0 or NDX_found==0):
            print('ERROR: NQ or NDX cannot find a time to align',date_i)
            return
        else:
            print('Adjust for date: ',date_i)
        while(NQ['datetime'].iloc[NQ_index] != NDX['datetime'].iloc[NDX_index]):
            print('Waring!')
            if(NQ['datetime'].iloc[NQ_index] > NDX['datetime'].iloc[NDX_index]):
                NDX_found=0
                time_destination2=NQ['date'].iloc[NQ_index].time()
                while(NQ['datetime'].iloc[NQ_index] < NDX['datetime'].iloc[NDX_index] and NDX_found==0 and NDX_index<NDX.shape[0] and NDX['date'].iloc[NDX_index].date()==date_i):
                    NDX_index=NDX_index+1
                    if(NDX['datetime'].iloc[NDX_index].time() >= time_destination2):
                        NDX_found=1
                        break
            else:
                NQ_found=0
                time_destination2=NDX['datetime'].iloc[NDX_index].time()
                while(NDX['datetime'].iloc[NDX_index] < NQ['datetime'].iloc[NQ_index] and NQ_found==0 and NQ_index<NQ.shape[0] and NQ['date'].iloc[NQ_index].date()==date_i):
                    NQ_index=NQ_index+1
                    if(NQ['datetime'].iloc[NQ_index].time() >= time_destination2):
                        NQ_found=1
                        break
            if(NQ_found==0 or NDX_found==0):
                print('ERROR: NQ or NDX cannot find a time to align after the time adjusted')
                return
        NDX_index2=NDX_index
        NQ_index2=NQ_index
        #k=NDX[column_name].iloc[NDX_index]/NQ[column_name].iloc[NQ_index]
        #k=NDX['close'].iloc[NDX_index]/NQ['close'].iloc[NQ_index]
        #print('K=', k)
        newdata=pd.DataFrame({'datetime': [day_i], 'NDX_index2': [NDX_index2], 'NQ_index2': [NQ_index2], 'time3': [NDX['datetime'].iloc[NDX_index]],'time4': [NQ['datetime'].iloc[NQ_index]]})
        list2=pd.concat([list2, newdata],ignore_index=True)
    list=pd.merge(list1, list2, on='datetime', how='left')
    list['d1']=list.apply(lambda row: NQ[column_name].iloc[int(row['NQ_index1'])] - NDX[column_name].iloc[int(row['NDX_index1'])], axis=1)
    list['d2']=list.apply(lambda row: NQ[column_name].iloc[int(row['NQ_index2'])] - NDX[column_name].iloc[int(row['NDX_index2'])], axis=1)
    list['dt']=list['time3']-list['time1']
    #print(list.head(10))
    #这一行之前都是用来制作list信息的，用以存放每一天的对齐时间和NQ的行数范围（从哪一行开始算这一天的）
    #通过制定的list的信息来进行缩放
    for day_i in range(list.shape[0]):
        if day_i != list.shape[0]-1:
               #NQ.loc[int(list['NQ_start'].iloc[day_i]):int(list['NQ_start'].iloc[day_i+1])-1, column_name]*=list['K'].iloc[day_i]
                NQ.iloc[int(list['NQ_start'].iloc[day_i]):int(list['NQ_start'].iloc[day_i+1]), NQ.columns.get_loc(column_name)] += (-1)*(list['d1'].iloc[day_i]+list['d2'].iloc[day_i])*0.5
        else:
                NQ.iloc[int(list['NQ_start'].iloc[day_i]):NQ.shape[0], NQ.columns.get_loc(column_name)] += (-1)*(list['d1'].iloc[day_i]+list['d2'].iloc[day_i])*0.5


def NQ_Align_A(QQQ,NQ):
    time1=datetime.time(15, 00)
    time2=datetime.time(16, 00)
    date_start=(NQ['datetime'].iloc[0]+datetime.timedelta(hours=6)).date()
    date_end=NQ['datetime'].iloc[-1].date()
    if(date_start != QQQ['datetime'].iloc[0].date()):
         print('ERROR: NQ and NDX start from differnt date')
         return
    if(date_end != QQQ['datetime'].iloc[-1].date()):
         print('ERROR: NQ and NDX end from differnt date')
         return
    day_num=(date_end-date_start).days
    NQ_index=0
    QQQ_index=0
    
    list= pd.DataFrame(columns=['datetime','K'])
    for day_i in range(day_num+1):
        date_i=date_start+datetime.timedelta(days=day_i)
        NQ_found=0
        QQQ_found=0
        while((NQ['datetime'].iloc[NQ_index]+datetime.timedelta(hours=6)).date()<date_i):
            NQ_index=NQ_index+1
        while(QQQ['datetime'].iloc[QQQ_index].date()<date_i):
            QQQ_index=QQQ_index+1
        NQ_start=NQ_index
        QQQ_start=QQQ_index
        if date_i.weekday() == 5 or date_i.weekday() == 6:
            continue
        newdata=pd.DataFrame({'datetime': [day_i], 'NQ_start': [int(NQ_start)],'QQQ_start': [int(QQQ_start)]})
        list=pd.concat([list, newdata],ignore_index=True)
    #print(list.head(10))
    for day_i in range(list.shape[0]):
        sum_NQ=0
        sum_QQQ=0
        if day_i != list.shape[0]-1:
            for i in range(int(list['NQ_start'].iloc[day_i]),int(list['NQ_start'].iloc[day_i+1])):
                if NQ['datetime'].iloc[i].time() >= time1 and NQ['datetime'].iloc[i].time() < time2:
                    sum_NQ += NQ['volume'].iloc[i]
            for i in range(int(list['QQQ_start'].iloc[day_i]),int(list['QQQ_start'].iloc[day_i+1])):
                if QQQ['datetime'].iloc[i].time() >= time1 and QQQ['datetime'].iloc[i].time() < time2:
                    sum_QQQ += QQQ['volume'].iloc[i]
        else:
            for i in range(int(list['NQ_start'].iloc[day_i]),NQ.shape[0]):
                if NQ['datetime'].iloc[i].time() >= time1 and NQ['datetime'].iloc[i].time() < time2:
                    sum_NQ += NQ['volume'].iloc[i]
            for i in range(int(list['QQQ_start'].iloc[day_i]),QQQ.shape[0]):
                if QQQ['datetime'].iloc[i].time() >= time1 and QQQ['datetime'].iloc[i].time() < time2:
                    sum_QQQ += QQQ['volume'].iloc[i]
        print('NQ:',sum_NQ)
        print('QQQ:',sum_QQQ)
        K=sum_QQQ/sum_NQ
        #print('K:',K)
        list.loc[day_i, 'K']=K
    print(list.head(10))
    #change it
    for day_i in range(list.shape[0]):
        if day_i != list.shape[0]-1:
                NQ.iloc[int(list['NQ_start'].iloc[day_i]):int(list['NQ_start'].iloc[day_i+1]), NQ.columns.get_loc('volume')] *= list['K'].iloc[int(day_i)]
        else:
                NQ.iloc[int(list['NQ_start'].iloc[day_i]):NQ.shape[0], NQ.columns.get_loc('volume')] *= list['K'].iloc[int(day_i)]

def NQ_Align(QQQ,NDX,NQ,timepoint1,timepoint2):
    NQ_Align_V('open',NDX,NQ,timepoint1,timepoint2)
    NQ_Align_V('close',NDX,NQ,timepoint1,timepoint2)
    NQ_Align_V('high',NDX,NQ,timepoint1,timepoint2)
    NQ_Align_V('low',NDX,NQ,timepoint1,timepoint2)
    NQ_Align_A(QQQ,NQ)
#NQ_Align("average",NDX,NQ,'120000')
#NQ_Align("open",NDX,NQ,'120000','140000')
#NQ_Align("high",NDX,NQ,'120000','140000')
#NQ_Align("low",NDX,NQ,'120000','140000')
#Close mast be the last aligned
#NQ_Align_V("close",NDX,NQ,'100000','150000')
#NQ_Align_A(QQQ,NQ)
#print(NDX.head())
#print(NQ.tail(10))
#print(QQQ.iloc[7400:7440].tail(10))
""" merged_list = pd.merge(QQQ, NQ, on='datetime', how='inner', suffixes=('_list1', '_list2'))
merged_list['ratio'] = merged_list['volume_list1'] / merged_list['volume_list2']
merged_list['ln_ratio']=0
print(merged_list.tail())
for i in range(merged_list.shape[0]): 
    merged_list['ln_ratio'].iloc[i] = np.log(merged_list['ratio'].iloc[i])
#print(NQ.tail())
# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(merged_list['ln_ratio'], marker='o', linestyle='-', color='b')

# 设置标题和标签
plt.title('Datetime vs ln_ratio')
plt.xlabel('Datetime')
plt.ylabel('ln_ratio')

# 旋转x轴标签以防重叠
plt.xticks(rotation=45)

# 显示网格
plt.grid(True)

# 显示图形
plt.show() """