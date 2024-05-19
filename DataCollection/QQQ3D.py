from ib_insync import *
import matplotlib as mpl
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd

from utils import HistoricalDataCollector
util.startLoop()  # uncomment this line when in a notebook
from config import parse_args
#Connection Establish
def main():

    args = parse_args()
    ib = IB()
    ib.connect('127.0.0.1', 7597, clientId=1)

    if args.contract_symbol == 'QQQ':
        args.secType = "STK"
        args.exchange = "SMART"
        args.currency = "USD"
    elif args.contract_symbol == 'NDX':
        args.secType = "IND"
        args.exchange = "NASDAQ"
        args.currency = "USD"
    elif args.contract_symbol == 'MNQ':
        args.secType = "FUT"
        args.exchange = "CME"
        args.currency = "USD"
        args.lastTradeDateOrContractMonth = "202409"  
    else:
        args.secType = "IND"
        args.exchange = "SMART"
        args.currency = "USD"       
    #Select Data Set
    DataCollector = HistoricalDataCollector(IBobject = ib,args = args)
    df = DataCollector.collect_historical_data(num_days=args.num_days)

    print(df.head(30))
    df.to_csv(f'{args.contract_symbol}_datatest.csv')

if __name__ == '__main__':
    main()