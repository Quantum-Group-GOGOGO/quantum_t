import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='quantum_t')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--date', type=str, default="20240501", help="Start date in YYYYMMDD format")

    parser.add_argument('--bar_size', default='1 min')
    parser.add_argument('--size', default = "1week" ,type=str,choices=["1week","1year"],help="number of days to download")
    parser.add_argument('--contract_symbol', default='QQQ',type=str,help="contract symbol,")

    parser.add_argument('--num_days', default = 10 ,type=int,help="number of days to download")
    parser.add_argument('--lastTradeDateOrContractMonth', default=None,type=str,help="YYYYMM,contract symbol,")

    args = parser.parse_args()
    return args