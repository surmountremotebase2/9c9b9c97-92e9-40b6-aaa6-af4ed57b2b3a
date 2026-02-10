from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import DCInsiderTrades
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [DCInsiderTrades()]
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
        dc_insider_trades_holdings = data[("dc_insider_trades",)]
        allocations = {"SPY": 1}
        if dc_insider_trades_holdings:
            alloc_dict = dc_insider_trades_holdings[-1]['allocations']
            log(f"Trading: {dc_insider_trades_holdings[-1]['allocations']}")
            allocations = alloc_dict
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)