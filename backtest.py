
import pandas as pd


def run_backtest(
    prices: pd.Series,
    fast_window: int = 20,
    slow_window: int = 100,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Compare a long/flat moving-average strategy with buy-and-hold.

    prices: historical prices.
    fast_window: the short moving-average window.
    slow_window: the long moving-average window.
    cost_bps: the assumed transaction cost in basis points.
    """
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Moving-average windows must be positive")

    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")

    prices = prices.dropna().astype(float).sort_index()

    if len(prices) <= slow_window:
        raise ValueError("Not enough prices for the slow moving average")

    if (prices <= 0).any():
        raise ValueError("Prices must be positive")

    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    df["market_return"] = prices.pct_change().fillna(0.0)

    df["fast_ma"] = prices.rolling(fast_window).mean()
    df["slow_ma"] = prices.rolling(slow_window).mean()

    signal = (df["fast_ma"] > df["slow_ma"]).astype(float)
    signal = signal.where(df["slow_ma"].notna(), 0.0)

    # The position is the signal shifted by one period to avoid lookahead bias.
    df["position"] = signal.shift(1).fillna(0.0)

    df["turnover"] = df["position"].diff().abs()
    if len(df):
        df.iloc[0, df.columns.get_loc("turnover")] = abs(
            df["position"].iloc[0]
        )

    cost_rate = cost_bps / 10_000.0
    df["strategy_return"] = (
        df["position"] * df["market_return"]
        - df["turnover"] * cost_rate
    )

    # Buy-and-hold starts with one unit of exposure.
    df["buy_hold_return"] = df["market_return"].copy()
    if len(df):
        df.iloc[0, df.columns.get_loc("buy_hold_return")] -= cost_rate

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = (1 + df["buy_hold_return"]).cumprod()

    return df
