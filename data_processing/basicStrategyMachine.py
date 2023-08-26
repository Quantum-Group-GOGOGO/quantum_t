import pandas as pd
import datetime as dt
import movingWindow as mw
from datetime import timedelta
from datetime import datetime
class BasicStrategyMachine:
    def __init__(self,dataWindow=None):
        self.data=dataWindow
        self.current_time = datetime.now()
        self.last_enter = datetime.now()
        self.status=0
    
    def setDataWindow(self,dataWindow):
        self.data=dataWindow
        self.current_time = dataWindow.date_last()
        self.last_enter = dataWindow.date_last()

    def answer(self):
        self.current_time = self.data.date_last()
        return False
    
#machine1=BasicStrategyMachine()
#machine2=BasicStrategyMachine()
#print(machine2.current_time.timestamp()-machine1.current_time.timestamp())