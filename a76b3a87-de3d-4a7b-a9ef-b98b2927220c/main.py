from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import CongressBuys
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [CongressBuys()]
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
        congress_buys_holdings = data[("congress_buys",)]
        
        if len(congress_buys_holdings) > 0:
            alloc_dict = congress_buys_holdings[-1]['allocations']
            #log(f"Trading: {congress_buys_holdings[-1]['allocations']}")
            allocations = alloc_dict
        else:
            allocations = {"SPY": 1}
            
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)