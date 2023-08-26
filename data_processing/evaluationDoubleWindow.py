from doubleWindow import DoubleWindow
from math import exp
class EvaluationDoubleWindow(DoubleWindow):
    def __init__(self,data,historyWindowSize=1,futureWindowSize=1):
        super().__init__(data,historyWindowSize,futureWindowSize)
        self._evaluationsize=self._futureWindowSize
        self._activation=1.0
        self._statistics._normalfactor=0.1
        self._statistics._value=0
        self.normalization()
        self.computeValue()

    def computeValue(self):
        result = 0.0
        for index in range(0,self._evaluationsize):
            result += (self.average(index)-self.average(0))*exp(index*(-self._activation))
        result = result / self._statistics._normalfactor / self.average(0)
        self._statistics._value = result
    
    def normalization(self):
        result = 0.0
        for index in range(0,self._evaluationsize):
            result += exp(index*(-self._activation))
        if result <= 0.0:
            result = 0.0000001
        self._statistics._normalfactor = result
    
    def setEvaluationSize(self,evaluationsize):
        if evaluationsize < 1:
            self._evaluationsize = 1
            self.normalization()
            return 1
        elif evaluationsize > self._futureWindowSize:
            self._evaluationsize = self._futureWindowSize
            self.normalization()
            return 2
        else:
            self._evaluationsize = evaluationsize
            self.normalization()
            return 0


    def setActivation(self,activation):
        if activation <= 0.0:
            self._activation=0.0
            self.normalization()
            return 1
        else:
            self._activation=activation
            self.normalization()
            return 0

    def init_evaluation(self,evaluationsize,activation):
        self.setEvaluationSize(evaluationsize)
        self.setActivation(activation)

    def value(self):
        self.computeValue()
        return self._statistics._value