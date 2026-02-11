from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import HouseLS
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [HouseLS()]
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
        house_ls_holdings = data[("house_ls",)]
        allocations = {"SPY": 1}
        if house_ls_holdings:
            alloc_dict = house_ls_holdings[-1]['allocations']
            log(f"Trading: {house_ls_holdings[-1]['allocations']}")
            allocations = alloc_dict
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)