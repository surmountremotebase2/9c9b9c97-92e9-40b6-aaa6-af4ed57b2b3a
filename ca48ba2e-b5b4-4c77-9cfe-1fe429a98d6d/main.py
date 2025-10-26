from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import InverseCramer
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [InverseCramer()]
        self.tickers = []

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        return self.tickers

    @property
    def data(self):
        return self.data_list

    def run(self, data):
        inverse_cramer_holdings = data[("inverse_cramer",)]
        allocations = {"AAPL": 1}
        if inverse_cramer_holdings:
            alloc_dict = inverse_cramer_holdings[-1]['allocations']
            #log(f"Trading: {inverse_cramer_holdings[-1]['allocations']}")
            allocations = alloc_dict
        #log(f"allocations:{allocations}")
        return TargetAllocation(allocations)