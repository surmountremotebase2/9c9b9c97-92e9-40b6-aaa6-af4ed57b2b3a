from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import LobbyQoQGrowth
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [LobbyQoQGrowth()]
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
        lobby_qoq_growth_holdings = data[("lobby_qoq_growth",)]
        allocations = {"SPY": 1}
        if len(lobby_qoq_growth_holdings) > 0:
            alloc_dict = lobby_qoq_growth_holdings[-1]['allocations']
            #log(f"Trading: {lobby_qoq_growth_holdings[-1]['allocations']}")
            allocations = alloc_dict

        #log(f"allocations:{allocations}")
        return TargetAllocation(allocations)