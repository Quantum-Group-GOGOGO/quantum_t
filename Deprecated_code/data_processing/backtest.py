class BackTest:
    def __init__(self,strategy,dataWindow):
        self.strategy=strategy
        self.strategy.setDataWindow(dataWindow)
        self.position=False
        self.earn=0.0
        self.principal=1.0
    
    def reinitial(self,enter,profit,lost):
        print("Initializing Strategy",enter,profit,lost)
        self.strategy.setEnterPoint(enter)
        self.strategy.setProfitExit(profit)
        self.strategy.setLostExit(lost)
        self.strategy.data.setWindowPosition(0)
        self.position=False
        self.earn=0.0

    def run(self):
        print("Running BackTest")
        print("")
        while not self.strategy.data.march():
            if self.strategy.answer():
                if self.position == False:
                    self.position = True
                else:
                    self.position = False
                    self.earn += (self.strategy.exit_price-self.strategy.enter_price)*self.principal/self.strategy.enter_price

    def getProfit(self):
        return self.earn
    
    def backTest(self,enter,profit,lost):
        self.reinitial(enter,profit,lost)
        self.run()
        return self.earn