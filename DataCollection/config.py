import argparse

""" def parse_args():  #Configs for NQ Data base reconstruction
    parser = argparse.ArgumentParser(description='quantum_t')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--date', type=str, default="20240621", help="Start date in YYYYMMDD format")

    parser.add_argument('--bar_size', default='1 min')
    parser.add_argument('--size', default = "1week" ,type=str,choices=["1week","1year"],help="number of days to download")
    parser.add_argument('--contract_symbol', default='NQ',type=str,help="contract symbol,")

    parser.add_argument('--num_days', default = 180 ,type=int,help="number of days to download")
    parser.add_argument('--lastTradeDateOrContractMonth', default=None,type=str,help="YYYYMM,contract symbol,")

    args = parser.parse_args()
    return args """

def parse_args():  #Configs for NQ Data base reconstruction
    parser = argparse.ArgumentParser(description='quantum_t')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--date', type=str, default="20241220", help="Start date in YYYYMMDD format")

    parser.add_argument('--bar_size', default='1 min')
    parser.add_argument('--size', default = "1week" ,type=str,choices=["1week","1year"],help="number of days to download")
    parser.add_argument('--contract_symbol', default='NQ',type=str,help="contract symbol,")

    parser.add_argument('--num_days', default = 1 ,type=int,help="number of days to download")
    parser.add_argument('--lastTradeDateOrContractMonth', default=None,type=str,help="YYYYMM,contract symbol,")

    args = parser.parse_args()
    return args