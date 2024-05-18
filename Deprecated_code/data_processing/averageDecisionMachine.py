import basicStrategyMachine as bsm
from datetime import datetime
from datetime import timedelta

class AverageDecisionMachine(bsm.BasicStrategyMachine):
    def __init__(self,dataWindow=None,enter_point=0.0,profit_exit=0.0,lost_exit=0.0,min_hold=60,max_hold=31556926):
        super().__init__(dataWindow)
        self.enter_point=enter_point
        self.profit_exit=profit_exit
        self.lost_exit=lost_exit
        self.enter_price=0.0
        self.exit_price=0.0
        self.min_hold=min_hold
        self.max_hold=max_hold
        self.current_time=0
        self.last_enter=self.current_time
        self.hold_second=self.current_time

    def setEnterPoint(slef,enter_point):
        slef.enter_point=enter_point
        return
    
    def setProfitExit(slef,profit_exit):
        slef.profit_exit=profit_exit
        return
    
    def setLostExit(slef,lost_exit):
        slef.lost_exit=lost_exit
        return
    
    def enterDecision(self):
        if (self.data.low_last()-self.data.getWindowMean())/self.data.getWindowMean()<self.enter_point:
            self.status=1
            self.last_enter = self.current_time
            self.enter_price = self.data.average_last()
            print("Enter in $",self.enter_price," Time=",self.data.date_last(),sep='')
            return 1
        return 0

    def exitDecision(self):
        if self.hold_second<self.min_hold:
            return 0
        if (self.data.low_last()-self.enter_price)/self.enter_price<self.lost_exit:
            self.status=0
            self.exit_price = self.data.average_last()
            print("L_Exit in $",self.exit_price," Time=",self.data.date_last(),sep='')
            return 1
        if (self.data.high_last()-self.enter_price)/self.enter_price>self.profit_exit:
            self.status=0
            self.exit_price = self.data.average_last()
            print("P_Exit in $",self.exit_price," Time=",self.data.date_last(),sep='')
            return 2
        if self.data.is_end() or self.hold_second>self.max_hold:
            self.status=0
            self.exit_price = self.data.average_last()
            print("T_Exit in $",self.exit_price," Time=",self.data.date_last(),sep='')
            return 3
        return 0 
    
    def answer(self):
        self.current_time = self.data.date_last()
        if self.status==0:
            return self.enterDecision()
        else:
            self.hold_second=(self.current_time-self.last_enter)// timedelta(seconds=1)
            return self.exitDecision()