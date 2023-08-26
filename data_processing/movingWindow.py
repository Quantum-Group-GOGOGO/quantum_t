import pandas as pd
import datetime as dt
from datetime import timedelta
class WindowStatistics:
     windowMean=0

class Window:
    def __init__(self,data,windowSize=1):
        if not isinstance(data,pd.DataFrame):
            raise Exception("Please Construct moving window with inserting pandas dataframe")
        if not 'date' in data.columns:
             raise Exception("Input Data does not have \"date\" column")
        if not 'high' in data.columns:
             raise Exception("Input Data does not have \"high\" column")
        if not 'low' in data.columns:
             raise Exception("Input Data does not have \"low\" column")
        if not 'volume' in data.columns:
             raise Exception("Input Data does not have \"volume\" column")
        if not 'average' in data.columns:
             raise Exception("Input Data does not have \"average\" column")
        self._data=data
        self._index=0
        self._windowSize=windowSize
        self._length=len(self._data)
        if self._length <1:
              raise Exception("DataFrame does not contain any data")
        self._endindex=self._index+self._windowSize-1
        self._secondLevelData=True
        if not data['average'].iloc[0] >= 0:
            self._secondLevelData=False
        self._timeInterval=(self._data['date'].iloc[1]-self._data['date'].iloc[0])/ timedelta(seconds=1)
        self._statistics=WindowStatistics()
        self._statistics.windowMean=self._meanCalculation()
    
    def setWindowSize(self,windowSize):
        if windowSize<1:
            self._windowSize=1
            self.refresh()
            return 1
        if self._index+windowSize>self._length:
            self._windowSize=self._length-self._index
            self.refresh()
            return 2
        self._windowSize=windowSize
        self.refresh()
        return 0
    
    def setWindowPosition(self,index):
        if index<0:
              self._index=0
              self.refresh()
              return 1
        if index+self._windowSize > self._length:
              self._index=self._length-self._windowSize
              self.refresh()
              return 2
        self._index=index
        self.refresh()
        return 0
    
    def march(self):
        if self._endindex>=self.length-1:
            return 1
        else:
            self._index += 1
            self._endindex += 1
            if self._secondLevelData==True:
                self._statistics.windowMean += (self._data['average'].iloc[self._endindex]-self._data['average'].iloc[self._index-1])/self._windowSize
                return 0
            else:
                self._statistics.windowMean += (self._data['high'].iloc[self._endindex]-self._data['high'].iloc[self._index-1])/self._windowSize/2
                self._statistics.windowMean += (self._data['low'].iloc[self._endindex]-self._data['low'].iloc[self._index-1])/self._windowSize/2
                return 0

    def b_march(self):
        if self._index<=0:
            return 1
        else:
            self._index -= 1
            self._endindex -= 1
            if self._secondLevelData==True:
                self._statistics.windowMean += (self._data['average'].iloc[self.index]-self._data['average'].iloc[self._endindex+1])/self._windowSize
                return 0
            else:
                self._statistics.windowMean += (self._data['high'].iloc[self.index]-self._data['high'].iloc[self._endindex+1])/self._windowSize/2
                self._statistics.windowMean += (self._data['low'].iloc[self.index]-self._data['low'].iloc[self._endindex+1])/self._windowSize/2
                return 0

    def refresh(self):
        self._endindex=self._index+self._windowSize-1
        self._statistics.windowMean=self._meanCalculation()

    def r_refresh(self):
        self._index=self._endindex-self._windowSize+1
        self._statistics.windowMean=self._meanCalculation()

    def average(self,searchIndex):
        if self._secondLevelData==True:
            if searchIndex >= self._windowSize:
                return self._data['average'].iloc[self._endindex]
            if searchIndex < 0:
                return self._data['average'].iloc[self._index]
            return self._data['average'].iloc[self._index+searchIndex]
        else:
            return (self.high(searchIndex)+self.low(searchIndex))*0.5
    
    def open(self,searchIndex):
        if searchIndex >= self._windowSize:
            return self._data['open'].iloc[self._endindex]
        if searchIndex < 0:
            return self._data['open'].iloc[self._index]
        return self._data['open'].iloc[self._index+searchIndex]

    
    def close(self,searchIndex):
        if searchIndex >= self._windowSize:
            return self._data['close'].iloc[self._endindex]
        if searchIndex < 0:
            return self._data['close'].iloc[self._index]
        return self._data['close'].iloc[self._index+searchIndex]
    
    def high(self,searchIndex):
        if searchIndex >= self._windowSize:
            return self._data['high'].iloc[self._endindex]
        if searchIndex < 0:
            return self._data['high'].iloc[self._index]
        return self._data['high'].iloc[self._index+searchIndex]
    
    def low(self,searchIndex):
        if searchIndex >= self._windowSize:
            return self._data['low'].iloc[self._endindex]
        if searchIndex < 0:
            return self._data['low'].iloc[self._index]
        return self._data['low'].iloc[self._index+searchIndex]
    
    def date(self,searchIndex):
        if searchIndex >= self._windowSize:
            return self._data['date'].iloc[self._endindex]
        if searchIndex < 0:
            return self._data['date'].iloc[self._index]
        return self._data['date'].iloc[self._index+searchIndex]
    
    def _meanCalculation(self):
        if self._secondLevelData==True:
            return self._data['average'].iloc[self._index:self._endindex].mean()
        else:
            return 0.5*(self._data['high'].iloc[self._index:self._endindex].mean()+self._data['low'].iloc[self._index:self._endindex].mean())
    
    def getWindowMean(self):
        return self._statistics.windowMean
    
    def open_last(self):
        return self.open(self._endindex)
    def close_last(self):
        return self.close(self._endindex)
    def high_last(self):
        return self.high(self._endindex)
    def low_last(self):
        return self.low(self._endindex)
    def average_last(self):
        return self.average(self._endindex)
    def date_last(self):
        return self.date(self._endindex)
    def is_end(self):
        if self._endindex >= self._length-1:
            return True
        return False

    @property
    def index(self):
         return self._index
    
    @property
    def windowSize(self):
         return self._windowSize
    
    @property
    def length(self):
         return self._length
    
    @property
    def endindex(self):
         return self._endindex
    
    @property
    def statistics(self):
         return self._statistics
    
