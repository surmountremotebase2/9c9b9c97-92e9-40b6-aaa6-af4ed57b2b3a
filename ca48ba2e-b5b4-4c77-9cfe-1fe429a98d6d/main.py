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
        # SPY price + 100 SMA
        # ----------------------
        ohlcv = data["ohlcv"]

        if len(ohlcv) < 100:
            log("Not enough data for 100-day SMA")
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

        base_alloc = {}
        if inverse_cramer_holdings:
            base_alloc = inverse_cramer_holdings[-1]["allocations"]

        final_alloc = {}

        if spy_above_sma:
            # Risk-on: +25% SPY
            final_alloc["SPY"] = 0.25
            remaining = 0.75
            regime = "SPY > 100 SMA → +25% SPY"

        else:
            # Defensive: +50% GLD
            final_alloc["GLD"] = 0.50
            remaining = 0.50
            regime = "SPY ≤ 100 SMA → +50% GLD"

        # ----------------------
        # Scale Inverse Cramer
        # ----------------------
        total_base = sum(base_alloc.values())
        if total_base > 0:
            for ticker, weight in base_alloc.items():
                final_alloc[ticker] = final_alloc.get(ticker, 0) + (
                    remaining * weight / total_base
                )

        log(f"Regime: {regime}")
        log(f"Final allocations: {final_alloc}")

        return TargetAllocation(final_alloc)