from surmount.base_class import Strategy, TargetAllocation
from surmount.data import InverseCramer
from surmount.logging import log
import pandas as pd

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [InverseCramer()]
        self.tickers = ["SPY", "GLD"]

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
        # ----------------------
        # SPY 100-day SMA
        # ----------------------
        ohlcv = data["ohlcv"]
        if len(ohlcv) < 100:
            return TargetAllocation({"SPY": 1})

        spy_close = pd.Series(
            [d["SPY"]["close"] for d in ohlcv],
            index=pd.to_datetime([d["SPY"]["date"] for d in ohlcv])
        )

        sma_100 = spy_close.rolling(100).mean()
        spy_above_sma = spy_close.iloc[-1] > sma_100.iloc[-1]

        # ----------------------
        # Regime allocation
        # ----------------------
        if spy_above_sma:
            regime_alloc = {"SPY": 0.25}
            stock_bucket = 0.75
            regime = "SPY > 100 SMA"
        else:
            regime_alloc = {"GLD": 0.50}
            stock_bucket = 0.50
            regime = "SPY ≤ 100 SMA"

        # ----------------------
        # Inverse Cramer (long-only)
        # ----------------------
        inverse_cramer_holdings = data[("inverse_cramer",)]
        base_alloc = inverse_cramer_holdings[-1]["allocations"] if inverse_cramer_holdings else {}

        longs = {k: v for k, v in base_alloc.items() if v > 0}
        long_total = sum(longs.values())

        final_alloc = dict(regime_alloc)

        if long_total > 0:
            for ticker, weight in longs.items():
                final_alloc[ticker] = stock_bucket * (weight / long_total)

        log(f"Regime: {regime}")
        log(f"Final allocations: {final_alloc}")

        return TargetAllocation(final_alloc)