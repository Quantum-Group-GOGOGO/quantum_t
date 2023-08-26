from movingWindow import Window
from movingWindow import WindowStatistics

class DoubleWindow:
    def __init__(self,data,historyWindowSize=1,futureWindowSize=1):
        self._data=data
        self._length=len(self._data)
        self._index=0
        self._beacon_index=historyWindowSize
        self._historyWindowSize=historyWindowSize
        self._futureWindowSize=futureWindowSize
        if historyWindowSize+futureWindowSize>self._length:
            raise Exception("Window size is larger than data size")
        self._historyWindow=Window(self._data,historyWindowSize)
        self._futureWindow=Window(self._data,futureWindowSize)
        self._futureWindow.setWindowPosition(self._beacon_index)
        self._statistics=WindowStatistics()

    def setHistoryWindowSize(self,windowSize):
        if windowSize<1:
            self._historyWindowSize=1
            self._historyWindow.setWindowSize(self._historyWindowSize)
            self._index=self._beacon_index-self._historyWindowSize
            self._historyWindow.setWindowPosition(self._index)
            self.refresh()
            return 1
        elif windowSize>self._beacon_index:
            self._historyWindowSize=self._beacon_index
            self._historyWindow.setWindowSize(self._historyWindowSize)
            self._index=self._beacon_index-self._historyWindowSize
            self._historyWindow.setWindowPosition(self._index)
            self.refresh()
            return 2
        else:
            self._historyWindowSize=windowSize
            self._historyWindow.setWindowSize(self._historyWindowSize)
            self._index=self._beacon_index-self._historyWindowSize
            self._historyWindow.setWindowPosition(self._index)
            self.refresh()
            return 0

    def setFutureWindowSize(self,windowSize):
        if windowSize<1:
            self._futureWindowSize=1
            self._futureWindow.setWindowSize(self._futureWindowSize)
            self.refresh()
            return 1
        elif windowSize>self._length-self._beacon_index:
            self._futureWindowSize=self._length-self._beacon_index
            self._futureWindow.setWindowSize(self._futureWindowSize)
            self.refresh()
            return 2
        else:
            self._futureWindowSize=windowSize
            self._futureWindow.setWindowSize(self._futureWindowSize)
            self.refresh()
            return 0

    def setWindowPosition(self,index):
        if index<0:
            self._index  =0
            self._beacon_index = self._historyWindowSize + self._index  
            self.refresh()
            return 1
        elif index + self._historyWindowSize + self._futureWindowSize > self._length:
            self._index = self._length-self._historyWindowSize + self._futureWindowSize
            self._beacon_index = self._historyWindowSize + self._index  
            self.refresh()
            return 2
        else:
            self._index=index
            self._beacon_index = self._historyWindowSize + self._index  
            self.refresh()
            return 0
    def setBeaconPosition(self,index):
    
    def march(self):
        if not self._futureWindow.march():
            self._historyWindow.march()
            self._index+=1
            self._beacon_index+=1
            return 0
        else:
            return 1

    def b_march(self):
        if not self._historyWindow.b_march():
            self._futureWindow.b_march()
            self._index-=1
            self._beacon_index-=1
            return 0
        else:
            return 1
        
    def refresh(self):
        self._historyWindow.refresh()
        self._futureWindow.refresh()
    def r_refresh(self):
        self._historyWindow.r_refresh()
        self._futureWindow.r_refresh()

    def average(self,searchIndex=0):
        if searchIndex<0:
            return self._historyWindow.average(self._historyWindowSize+searchIndex)
        else:
            return self._futureWindow.average(searchIndex)
    
    def open(self,searchIndex=0):
        if searchIndex<0:
            return self._historyWindow.open(self._historyWindowSize+searchIndex)
        else:
            return self._futureWindow.open(searchIndex)
        
    def close(self,searchIndex=0):
        if searchIndex<0:
            return self._historyWindow.close(self._historyWindowSize+searchIndex)
        else:
            return self._futureWindow.close(searchIndex)
        
    def high(self,searchIndex=0):
        if searchIndex<0:
            return self._historyWindow.high(self._historyWindowSize+searchIndex)
        else:
            return self._futureWindow.high(searchIndex)
        
    def low(self,searchIndex=0):
        if searchIndex<0:
            return self._historyWindow.low(self._historyWindowSize+searchIndex)
        else:
            return self._futureWindow.low(searchIndex)
        
    def date(self,searchIndex=0):
        if searchIndex<0:
            return self._historyWindow.date(self._historyWindowSize+searchIndex)
        else:
            return self._futureWindow.date(searchIndex)

    def is_end(self):
        return self._futureWindow.is_end()