
import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from metrics import calculate_metrics


def sample_prices(n=150):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    values = np.linspace(100, 150, n)
    return pd.Series(values, index=dates, name="close")


def test_backtest_returns_expected_columns():
    result = run_backtest(sample_prices())

    assert "strategy_return" in result.columns
    assert "buy_hold_return" in result.columns
    assert "strategy_equity" in result.columns


def test_equity_curve_starts_at_one():
    result = run_backtest(sample_prices(), cost_bps=0)

    # Equity starts at one when the first modeled return is zero.
    assert result["buy_hold_equity"].iloc[0] == pytest.approx(1.0)


def test_fast_window_must_be_less_than_slow_window():
    with pytest.raises(ValueError):
        run_backtest(sample_prices(), fast_window=100, slow_window=20)


def test_negative_transaction_cost_is_rejected():
    with pytest.raises(ValueError):
        run_backtest(sample_prices(), cost_bps=-1)


def test_metrics_include_max_drawdown():
    returns = pd.Series([0.01, 0.02, -0.10, 0.01])
    metrics = calculate_metrics(returns)

    assert "max_drawdown" in metrics
    assert metrics["max_drawdown"] < 0
