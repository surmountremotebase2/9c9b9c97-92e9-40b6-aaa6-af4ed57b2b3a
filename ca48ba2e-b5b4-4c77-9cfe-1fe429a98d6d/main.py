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
        # Inverse Cramer
        # ----------------------
        inverse_cramer_holdings = data[("inverse_cramer",)]
        base_alloc = inverse_cramer_holdings[-1]["allocations"] if inverse_cramer_holdings else {}

        final_alloc = {}

        # ----------------------
        # Regime asset
        # ----------------------
        if spy_above_sma:
            final_alloc["SPY"] = 0.25
            regime = "SPY > 100 SMA"
        else:
            final_alloc["GLD"] = 0.50
            regime = "SPY ≤ 100 SMA"

        # ----------------------
        # Convert long/short → long-only
        # ----------------------
        long_total = sum(w for w in base_alloc.values() if w > 0)
        short_abs_total = sum(abs(w) for w in base_alloc.values() if w < 0)

        # Add long positions
        for ticker, weight in base_alloc.items():
            if weight > 0:
                final_alloc[ticker] = final_alloc.get(ticker, 0) + weight

        # Redirect short exposure to SPY
        final_alloc["SPY"] = final_alloc.get("SPY", 0) + short_abs_total

        # ----------------------
        # Normalize to 100%
        # ----------------------
        total = sum(final_alloc.values())
        if total > 0:
            for ticker in final_alloc:
                final_alloc[ticker] /= total

        log(f"Regime: {regime}")
        log(f"Final allocations: {final_alloc}")

        return TargetAllocation(final_alloc)