from surmount.base_class import Strategy, TargetAllocation
from surmount.data import CongressBuys
from surmount.logging import log
import pandas as pd

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [CongressBuys()]
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
        # Get SPY price history
        # ----------------------
        ohlcv = data["ohlcv"]

        if len(ohlcv) < 10:
            log("Not enough data for 200-day SMA")
            return TargetAllocation({"SPY": 1})

        spy_close = pd.Series(
            [d["SPY"]["low"] for d in ohlcv],
            index=pd.to_datetime([d["SPY"]["date"] for d in ohlcv])
        )

        sma_200 = spy_close.rolling(100).mean()
        spy_above_sma = spy_close.iloc[-1] > sma_200.iloc[-1]

        # ----------------------
        # Congress Buys
        # ----------------------
        congress_buys_holdings = data[("congress_buys",)]

        congress_alloc = {}
        if len(congress_buys_holdings) > 0:
            congress_alloc = congress_buys_holdings[-1]["allocations"]

        final_alloc = {}

        if spy_above_sma:
            # 50% SPY
            final_alloc["SPY"] = 0.25

            # Remaining 50% to Congress buys (scaled)
            total_cb = sum(congress_alloc.values())
            if total_cb > 0:
                for ticker, weight in congress_alloc.items():
                    final_alloc[ticker] = 0.75 * (weight / total_cb)

            regime = "Risk ON (SPY > 200 SMA)"

        else:
            final_alloc["GLD"] = 0.5
            # No SPY, Congress buys only up to 50%
            total_cb = sum(congress_alloc.values())
            if total_cb > 0:
                for ticker, weight in congress_alloc.items():
                    final_alloc[ticker] = 0.50 * (weight / total_cb)

            regime = "Risk OFF (SPY ≤ 200 SMA)"

        log(f"Regime: {regime}")
        log(f"Final allocation: {final_alloc}")

        return TargetAllocation(final_alloc)