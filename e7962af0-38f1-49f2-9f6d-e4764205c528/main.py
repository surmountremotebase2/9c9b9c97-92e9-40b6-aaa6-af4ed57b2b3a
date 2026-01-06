from surmount.base import Strategy, TargetAllocation
import pandas as pd
import numpy as np


class TradingStrategy(Strategy):
    """
    Jason-style blended momentum strategy:
    - 75% short-term (weekly proxy)
    - 25% long-term (monthly proxy)
    - Keltner channel on Score
    - Ichimoku regime filter (pass/fail)
    """

    def __init__(self):
        self.assets = [
            "SPY", "QQQ", "TLT", "IEF", "AGG", "BIL"
        ]

    @staticmethod
    def tsi(series, short, long):
        diff = series.diff()
        abs_diff = diff.abs()

        num = diff.ewm(span=short).mean().ewm(span=long).mean()
        den = abs_diff.ewm(span=short).mean().ewm(span=long).mean()

        return num / den

    def compute_score(self, prices):
        # Short (weekly proxy)
        tsi_short = self.tsi(prices, short=10, long=20)

        # Long (monthly proxy)
        tsi_long = self.tsi(prices, short=40, long=80)

        # Blended score
        score = 0.75 * tsi_short + 0.25 * tsi_long
        return score

    def keltner_midline(self, score, length=31):
        return score.rolling(length).mean()

    def ichimoku_pass(self, prices):
        # Simplified Ichimoku pass/fail
        high = prices.rolling(52).max()
        low = prices.rolling(52).min()
        cloud_mid = (high + low) / 2
        return prices.iloc[-1] > cloud_mid.iloc[-1]

    def run(self, data):
        scores = {}
        latest_scores = {}

        for asset in self.assets:
            prices = data[asset]["close"]

            if len(prices) < 100:
                continue

            score = self.compute_score(prices)
            midline = self.keltner_midline(score)

            # Keltner trend condition
            trend_ok = score.iloc[-1] > midline.iloc[-1]

            # Ichimoku regime filter
            regime_ok = self.ichimoku_pass(prices)

            if trend_ok and regime_ok:
                scores[asset] = score.iloc[-1]

        if not scores:
            return TargetAllocation({"BIL": 1.0})

        # Rank assets by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Allocate equally to top N assets
        top_n = min(2, len(ranked))
        weight = 1.0 / top_n

        allocation = {
            asset: weight for asset, _ in ranked[:top_n]
        }

        return TargetAllocation(allocation)