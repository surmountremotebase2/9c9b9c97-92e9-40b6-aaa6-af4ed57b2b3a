from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import RobBresnahan
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [RobBresnahan()]
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
        rob_bresnahan_holdings = data[("rob_bresnahan",)]
        allocations = {"SPY": 1}
        if rob_bresnahan_holdings:
            alloc_dict = rob_bresnahan_holdings[-1]['allocations']
            log(f"Trading: {rob_bresnahan_holdings[-1]['allocations']}")
            allocations = alloc_dict
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)