
import numpy as np
import pandas as pd


def calculate_metrics(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Calculate common performance metrics from periodic returns."""
    returns = returns.dropna().astype(float)

    if returns.empty:
        raise ValueError("Cannot calculate metrics on empty returns")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    years = len(returns) / periods_per_year

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 and equity.iloc[-1] > 0
        else float("nan")
    )

    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    sharpe = (
        mean_return / std_return * np.sqrt(periods_per_year)
        if pd.notna(std_return) and std_return > 0
        else float("nan")
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
