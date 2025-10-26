from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import TimMoore, AnalystLong
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [AnalystLong()]
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
        tim_moore_holdings = data[("analyst_long",)]
        allocations = {"AAPL":1}      
        if tim_moore_holdings:
            alloc_dict = tim_moore_holdings[-1]['allocations']
            log(f"Trading: {tim_moore_holdings[-1]['allocations']}")
            allocations = alloc_dict
            #allocations =  dict(list(alloc_dict.items())[-5:])
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)