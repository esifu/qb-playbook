
import yfinance as yf
import pandas as pd


def load_prices(
    ticker: str = "SPY",
    start: str = "2000-01-01",
    end: str = "2026-10-27",
) -> pd.Series:
    """
    Download daily prices for an ETF and return its adjusted close.

    Fixed start and end dates keep runs on the same data range.
    end is exclusive, matching yfinance.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No market data returned for {ticker}")

    # Recent yfinance versions may return MultiIndex columns.
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        else:
            close = close.iloc[:, 0]

    close = close.dropna().astype(float)
    close.name = "close"

    if close.empty:
        raise ValueError(f"No valid closing prices for {ticker}")

    return close
