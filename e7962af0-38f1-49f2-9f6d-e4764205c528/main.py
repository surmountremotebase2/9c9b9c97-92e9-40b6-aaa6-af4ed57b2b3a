import pandas as pd
import numpy as np
from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


class TradingStrategy(Strategy):
    """
    Jason Lipps Momentum Strategy (Surmount-compatible)

    - Blended TSI Score:
        75% short-term (weekly proxy)
        25% long-term (monthly proxy)
    - Keltner Channel applied to Score
    - Ichimoku Cloud as pass/fail regime filter
    - Binary risk-on / risk-off allocation
    """

    def __init__(self):
        self._assets = ["SPY", "BIL"]
        self.rebalance_day = 1  # Tuesday
        self.last_alloc = {"SPY": 0.0, "BIL": 1.0}
        self.score_history = []

    @property
    def assets(self):
        return self._assets

    @property
    def interval(self):
        return "1day"

    # --------------------
    # Indicator helpers
    # --------------------

    def tsi(self, close, short, long):
        diff = close.diff()
        abs_diff = diff.abs()

        num = diff.ewm(span=short).mean().ewm(span=long).mean()
        den = abs_diff.ewm(span=short).mean().ewm(span=long).mean()

        return num / den

    def ichimoku_pass(self, close):
        high_52 = close.rolling(52).max()
        low_52 = close.rolling(52).min()
        cloud_mid = (high_52 + low_52) / 2
        return close.iloc[-1] > cloud_mid.iloc[-1]

    # --------------------
    # Main execution
    # --------------------

    def run(self, data):
        ohlcv = data["ohlcv"]

        # Warmup
        if len(ohlcv) < 120:
            return TargetAllocation(self.last_alloc)

        # Build SPY close series
        spy_close = pd.Series(
            [d["SPY"]["close"] for d in ohlcv],
            index=pd.to_datetime([d["SPY"]["date"] for d in ohlcv])
        )

        today = spy_close.index[-1]
        if today.weekday() != self.rebalance_day:
            return TargetAllocation(self.last_alloc)

        # --------------------
        # Score computation
        # --------------------

        tsi_short = self.tsi(spy_close, short=10, long=20)
        tsi_long = self.tsi(spy_close, short=40, long=80)

        score = 0.75 * tsi_short + 0.25 * tsi_long
        self.score_history.append(score.iloc[-1])

        # Keltner channel on Score
        score_series = pd.Series(self.score_history)
        midline = score_series.rolling(31).mean()

        if len(midline.dropna()) == 0:
            return TargetAllocation(self.last_alloc)

        trend_ok = score_series.iloc[-1] > midline.iloc[-1]
        regime_ok = self.ichimoku_pass(spy_close)

        # --------------------
        # Allocation logic
        # --------------------

        if trend_ok and regime_ok:
            alloc = {"SPY": 1.0, "BIL": 0.0}
        else:
            alloc = {"SPY": 0.0, "BIL": 1.0}

        self.last_alloc = alloc
        log(f"Score={round(score.iloc[-1], 3)} | SPY={alloc['SPY']}")

        return TargetAllocation(self.last_alloc)