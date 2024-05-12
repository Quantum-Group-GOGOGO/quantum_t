from ib_insync import *
class DatacollectionQQQ1Day:
    def _init_(self):
        return
    def DatacollectionQQQ1Day(IBobject, date, barSize):
        contract = Contract()
        contract  = Stock('QQQ','SMART','USD')
        bars = IBobject.reqHistoricalData(
        contract, endDateTime=(date+' 00:00:00'), durationStr='1 D',
        barSizeSetting=barSize, whatToShow='TRADES', useRTH=False)

        df = util.df(bars)
        return df