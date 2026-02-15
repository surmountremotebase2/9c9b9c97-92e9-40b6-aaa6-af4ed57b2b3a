from strategies.BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd
import os


class IchiCloudMom(BaseStrategy):
    """
    Jason Lipps Momentum Strategy – Full Feature Implementation

    Features:
    - Blended TSI score (75% weekly proxy, 25% monthly proxy)
    - Score smoothing + Score ROC
    - Keltner Channel on Score (31, 4x)
    - Static score support/resistance (blue lines proxy)
    - Ichimoku Cloud pass/fail
    - Cross-ETF ranking
    - 25 / 50 / 100 exposure scaling
    - Safe asset fallback (BIL)
    """

    def __init__(self):
        super().__init__(
            strategy_name="IchiCloudMom",
            strategy_version="3.0",
            MacroTickers=["MEDCPIM158SFRBCLE"],
            tickers=["SPY", "QQQ", "TLT", "IEF", "BIL"]
        )

        self.risk_assets = ["SPY", "QQQ", "TLT", "IEF"]
        self.safe_asset = "BIL"
        self.Benchmark = "SPY"
        self.CPI = "MEDCPIM158SFRBCLE"

        self.WARMUP = 252
        self.strategy_file = os.path.basename(__file__)

        self.last_alloc = {a: 0.0 for a in self.risk_assets + [self.safe_asset]}
        self.last_alloc[self.safe_asset] = 1.0

        self.score_hist = {a: [] for a in self.risk_assets}

    # -------------------------------------------------
    # Indicators
    # -------------------------------------------------
    def tsi(self, close, short, long):
        diff = close.diff()
        abs_diff = diff.abs()
        num = diff.ewm(span=short).mean().ewm(span=long).mean()
        den = abs_diff.ewm(span=short).mean().ewm(span=long).mean()
        return num / den

    def ichimoku_base(self, high, low):
        return (high.rolling(26).max() + low.rolling(26).min()) / 2

    def score_support(self, score_series):
        """
        Proxy for Jason's manual blue support lines:
        Use lower-quantile cluster of historical scores.
        """
        if len(score_series) < 100:
            return np.nan
        return score_series.quantile(0.20)

    # -------------------------------------------------
    # Main execution
    # -------------------------------------------------
    def run_strategy(self, data, MacroData):
        data = data.ffill().bfill()

        allocation = pd.DataFrame(
            index=data.index,
            columns=self.risk_assets + [self.safe_asset],
            dtype=float
        ).fillna(0.0)

        allocation.iloc[0][self.safe_asset] = 1.0

        for i in range(1, len(data)):
            allocation.iloc[i] = allocation.iloc[i - 1]

            if i < self.WARMUP:
                continue

            date = data.index[i]

            scores = {}
            score_roc = {}
            regimes = {}

            # -------------------------------
            # Score computation per asset
            # -------------------------------
            for asset in self.risk_assets:
                close = data["Close"][asset].iloc[:i]
                high = data["High"][asset].iloc[:i]
                low = data["Low"][asset].iloc[:i]

                tsi_short = self.tsi(close, 10, 20).iloc[-1]
                tsi_long = self.tsi(close, 40, 80).iloc[-1]

                raw_score = 0.75 * tsi_short + 0.25 * tsi_long
                self.score_hist[asset].append(raw_score)

                score_series = pd.Series(self.score_hist[asset])
                smooth_score = score_series.rolling(5).mean().iloc[-1]

                if np.isnan(smooth_score):
                    continue

                scores[asset] = smooth_score
                score_roc[asset] = score_series.diff(5).iloc[-1]

                ichi_base = self.ichimoku_base(high, low).iloc[-1]
                regimes[asset] = close.iloc[-1] > ichi_base

            if not scores:
                allocation.iloc[i][self.safe_asset] = 1.0
                continue

            # -------------------------------
            # Rank assets (Score + ROC)
            # -------------------------------
            ranked = sorted(
                scores.keys(),
                key=lambda a: (scores[a], score_roc[a]),
                reverse=True
            )

            top = ranked[0]
            score_series = pd.Series(self.score_hist[top])

            # -------------------------------
            # Keltner on Score
            # -------------------------------
            midline = score_series.rolling(31).mean().iloc[-1]
            lower_band = midline - 4 * score_series.rolling(31).std().iloc[-1]
            support = self.score_support(score_series)

            if np.isnan(midline) or np.isnan(lower_band):
                continue

            # -------------------------------
            # Exposure scaling
            # -------------------------------
            exposure = 0.0
            if regimes[top]:
                if scores[top] > midline:
                    exposure = 1.0
                elif scores[top] > lower_band:
                    exposure = 0.50
                elif scores[top] > support:
                    exposure = 0.25

            alloc = {a: 0.0 for a in allocation.columns}
            alloc[top] = exposure
            alloc[self.safe_asset] = 1.0 - exposure

            allocation.loc[date] = alloc
            self.last_alloc = alloc

        return allocation