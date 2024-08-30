import pandas as pd
from datetime import datetime, timedelta
from ib_insync import *
from tqdm import tqdm
class HistoricalDataCollector:
    def __init__(self, IBobject, args):
        self.IBobject = IBobject
        self.start_date = datetime.strptime(args.date, '%Y%m%d').date()
        self.bar_size = args.bar_size
        self.contract_symbol = args.contract_symbol
        self.secType = args.secType
        self.exchange = args.exchange
        self.currency = args.currency
        self.lastTradeDateOrContractMonth = args.lastTradeDateOrContractMonth if args.lastTradeDateOrContractMonth else None

    @staticmethod
    def is_weekend(date_str):
        date_format = '%Y%m%d'
        date = datetime.strptime(date_str, date_format).date()
        return date.weekday() == 5 or date.weekday() == 6

    def _create_contract(self):
        contract = Contract()
        contract.symbol = self.contract_symbol
        contract.secType = self.secType
        contract.exchange = self.exchange
        contract.currency = self.currency
        if self.lastTradeDateOrContractMonth:
            contract.lastTradeDateOrContractMonth = self.lastTradeDateOrContractMonth
        return contract

    def _req_historical_data_for_date(self, date_str):
        if not self.is_weekend(date_str):  # Ensure it's not a weekend before requesting data
            date = datetime.strptime(date_str, '%Y%m%d').date()
            # Adjust the format of endDateTime to include spaces between date, time, and timezone (assuming local timezone)
            formatted_end_date = f'{date.strftime("%Y%m%d")} {date.strftime("%H:%M:%S")}'
            contract = self._create_contract()
            bars = self.IBobject.reqHistoricalData(
                contract=contract,
                endDateTime=formatted_end_date,  # Updated format here
                durationStr='1 D',
                barSizeSetting=self.bar_size,
                whatToShow='TRADES',
                useRTH=False
            )
            return util.df(bars)
        return None  # Return None if it's a weekend day

    def collect_historical_data(self, num_days):
        dfs = []
        for i in tqdm(range(num_days)):
            date_str = (self.start_date - timedelta(days=i)).strftime('%Y%m%d')
            df = self._req_historical_data_for_date(date_str)
            if df is not None:  # Only append if data was collected (not a weekend)
                dfs.append(df)
        try:
            # import pdb;pdb.set_trace()
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset='date')
            combined_df =  combined_df.sort_values(by='date', ascending=False,inplace=False)  
        except ValueError:  # Handle the case where no data was collected
            combined_df = pd.DataFrame()
        return combined_df

# Usage remains the same
# args = ...  # Define your arguments here
# ib_object = ...  # Initialize your Interactive Brokers API object
# collector = HistoricalDataCollector(ib_object, args)
# historical_data = collector.collect_historical_data(10)  # Collect data for the past 10 days