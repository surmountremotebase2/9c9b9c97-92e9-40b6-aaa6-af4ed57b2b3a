from surmount.base_class import Strategy, TargetAllocation
from surmount.data import CongressBuys, OHLCV
from surmount.logging import log
import numpy as np

class TradingStrategy(Strategy):
    def __init__(self):
        self.data_list = [
            CongressBuys(),
            OHLCV("SPY")
        ]
        self.tickers = ["SPY"]

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
        congress_buys = data[("congress_buys",)]
        spy_ohlc = data[("ohlcv", "SPY")]

        # --- Compute 200-day SMA ---
        if len(spy_ohlc) < 200:
            log("Not enough SPY data for 200-day SMA")
            return TargetAllocation({"SPY": 1.0})

        closes = np.array([bar["close"] for bar in spy_ohlc])
        sma_200 = closes[-200:].mean()
        spy_price = closes[-1]

        spy_above_sma = spy_price > sma_200
        log(f"SPY: {spy_price:.2f}, SMA200: {sma_200:.2f}, Above SMA: {spy_above_sma}")

        allocations = {}

        # --- Congress allocations ---
        if len(congress_buys) > 0:
            congress_alloc = congress_buys[-1]["allocations"]
        else:
            congress_alloc = {}

        if spy_above_sma:
            # 50% SPY + 50% Congress stocks
            allocations["SPY"] = 0.5

            total_congress = sum(congress_alloc.values())
            if total_congress > 0:
                for k, v in congress_alloc.items():
                    allocations[k] = 0.5 * (v / total_congress)
        else:
            # Only Congress stocks, capped at 50%
            total_congress = sum(congress_alloc.values())
            if total_congress > 0:
                for k, v in congress_alloc.items():
                    allocations[k] = 0.5 * (v / total_congress)

        log(f"Final allocations: {allocations}")
        return TargetAllocation(allocations)