from surmount.base_class import Strategy, TargetAllocation, backtest
from surmount.data import InsiderPurchasesMin500MMarketCap
from surmount.logging import log

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [InsiderPurchasesMin500MMarketCap()]
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
        insider_purchases_min_500m_market_cap_holdings = data[("insider_purchases_min_500m_market_cap",)]
        allocations = {"SPY": 1}
        if insider_purchases_min_500m_market_cap_holdings:
            alloc_dict = insider_purchases_min_500m_market_cap_holdings[-1]['allocations']
            log(f"Trading: {insider_purchases_min_500m_market_cap_holdings[-1]['allocations']}")
            allocations = alloc_dict
        log(f"allocations:{allocations}")
        return TargetAllocation(allocations)