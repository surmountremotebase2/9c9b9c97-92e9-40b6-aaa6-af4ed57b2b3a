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
        allocations = {"BIL": 1}
        if congress_buys_holdings:
            alloc_dict = congress_buys_holdings[-1]['allocations']
            #log(f"Trading: {congress_buys_holdings[-1]['allocations']}")
            allocations = alloc_dict
            # If BBY is in the allocation, move its weight to BIL
            if "BBY" in alloc_dict:
                weight = alloc_dict.pop("BBY")      # remove BBY and get its allocation
                alloc_dict["BIL"] = weight         # set BIL to that allocation
            if "HY" in alloc_dict:
                weight = alloc_dict.pop("HY")
                alloc_dict["BIL"] = weight
            if "MS-P" in alloc_dict:
                weight = alloc_dict.pop("MS-P")
                alloc_dict["BIL"] = weight
            if "DTM" in alloc_dict:
                weight = alloc_dict.pop("DTM")
                alloc_dict["BIL"] = weight
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)