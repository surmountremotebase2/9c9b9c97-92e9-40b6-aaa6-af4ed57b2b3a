import pandas as pd
import numpy as np
from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


class TradingStrategy(Strategy):
    """
    Jason Lipps Multi-Asset Momentum Strategy
    ------------------------------------------
    Core Features:
    - TSI(10) computed on:
        • Weekly candles (short-term component)
        • Monthly candles (long-term component)
    - Blended Score = 75% Weekly + 25% Monthly
    - 5-period smoothing of Score
    - Keltner channel (31, 4×vol proxy) on Score
    - Ichimoku Base line pass/fail regime filter
    - Cross-asset ranking
    - 25% / 50% / 100% exposure scaling
    - Weekly rebalance
    """

    def __init__(self):
        self._assets = ["SPY", "QQQ", "TLT", "IEF", "BIL"]
        self.risk_assets = ["SPY", "QQQ", "TLT", "IEF"]
        self.safe_asset = "BIL"

        self.rebalance_day = 1  # Tuesday
        self.last_alloc = {a: 0.0 for a in self._assets}
        self.last_alloc[self.safe_asset] = 1.0

    @property
    def assets(self):
        return self._assets

    @property
    def interval(self):
        return "1day"

    # -------------------------------------------------
    # Indicator Helpers
    # -------------------------------------------------

    def tsi(self, close, period=10):
        diff = close.diff()
        abs_diff = diff.abs()

        ema1 = diff.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()

        abs1 = abs_diff.ewm(span=period).mean()
        abs2 = abs1.ewm(span=period).mean()

        return ema2 / abs2

    def ichimoku_base(self, high, low, period=26):
        return (high.rolling(period).max() +
                low.rolling(period).min()) / 2

    def keltner_score(self, score_series):
        mid = score_series.rolling(100).mean()
        vol = score_series.rolling(100).std()
        lower = mid - 4 * vol
        return mid.iloc[-1], lower.iloc[-1]

    # -------------------------------------------------
    # Main Execution
    # -------------------------------------------------

    def run(self, data):

        ohlcv = data["ohlcv"]
        if len(ohlcv) < 1:
            return TargetAllocation(self.last_alloc)

        today = pd.to_datetime(ohlcv[-1]["SPY"]["date"])
        if today.weekday() != self.rebalance_day:
            return TargetAllocation(self.last_alloc)

        asset_scores = {}

        for asset in self.risk_assets:

            df = pd.DataFrame({
                "close": [d[asset]["close"] for d in ohlcv],
                "high":  [d[asset]["high"] for d in ohlcv],
                "low":   [d[asset]["low"] for d in ohlcv],
            }, index=pd.to_datetime([d[asset]["date"] for d in ohlcv]))

            # --- Weekly Resample ---
            weekly = df.resample("W-FRI").last()
            weekly_tsi = self.tsi(weekly["close"], period=10)

            # --- Monthly Resample ---
            monthly = df.resample("M").last()
            monthly_tsi = self.tsi(monthly["close"], period=10)

            if len(weekly_tsi.dropna()) < 5 or len(monthly_tsi.dropna()) < 3:
                continue

            score = 0.75 * weekly_tsi.iloc[-1] + 0.25 * monthly_tsi.iloc[-1]

            # 5-period smoothing (weekly equivalent)
            score_series = weekly_tsi.dropna().rolling(5).mean()
            score_smoothed = 0.75 * score_series.iloc[-1] + 0.25 * monthly_tsi.iloc[-1]

            # Score ROC (durability)
            score_roc = (score_smoothed - score_series.iloc[-5]) * 100
            log(f"Asset : {asset} | Score : {score_roc}")


            # Ichimoku pass/fail
            base_line = self.ichimoku_base(df["high"], df["low"])
            regime_pass = df["close"].iloc[-1] > base_line.iloc[-1]

            asset_scores[asset] = {
                "score": score_smoothed,
                "roc": score_roc,
                "regime": regime_pass,
                "history": score_series
            }

        if len(asset_scores) == 0:
            return TargetAllocation(self.last_alloc)

        # Filter regime
        candidates = {k: v for k, v in asset_scores.items() if v["regime"]}

        if len(candidates) == 0:
            alloc = {a: 0.0 for a in self._assets}
            alloc[self.safe_asset] = 1.0
            self.last_alloc = alloc
            return TargetAllocation(self.last_alloc)

        # Rank by Score then ROC
        ranked = sorted(
            candidates.items(),
            key=lambda x: (x[1]["score"], x[1]["roc"]),
            reverse=True
        )

        top_asset, top_data = ranked[0]

        # Keltner logic
        mid, lower = self.keltner_score(top_data["history"])

        exposure = 0.0
        if top_data["score"] > mid:
            exposure = 1.0
        elif top_data["score"] > lower:
            exposure = 0.5
        elif top_data["score"] > lower * 0.8:
            exposure = 0.25
        else:
            exposure = 0.0

        alloc = {a: 0.0 for a in self._assets}
        alloc[top_asset] = float(exposure)
        alloc[self.safe_asset] = float(1.0 - exposure)

        self.last_alloc = alloc

        log(f"Top Asset: {top_asset} | Exposure: {exposure}")

        return TargetAllocation(self.last_alloc)