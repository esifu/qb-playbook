from pathlib import Path

import pandas as pd

from data_loader import load_prices
from backtest import run_backtest
from metrics import calculate_metrics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TICKER = "SPY"
PERIOD = "10y"
COST_BPS = 5.0

# Each tuple is (fast_window, slow_window).
PARAMETER_COMBINATIONS = [
    (10, 50),
    (20, 100),
    (30, 150),
    (10, 100),
    (20, 150),
]

OUTPUT_DIR = Path("results")
OUTPUT_FILE = OUTPUT_DIR / "parameter_sensitivity.csv"


# --------------------------------------------------
# Run parameter sensitivity analysis
# --------------------------------------------------

def main():
    print(f"Loading historical prices for {TICKER}...")

    prices = load_prices(ticker=TICKER, period=PERIOD)

    if prices.empty:
        raise ValueError(f"No price data available for {TICKER}.")

    results = []

    # Calculate buy-and-hold metrics once as a benchmark.
    baseline = run_backtest(
        prices,
        fast_window=20,
        slow_window=100,
        cost_bps=COST_BPS,
    )

    benchmark_metrics = calculate_metrics(
        baseline["buy_hold_return"]
    )

    print(f"\nTesting {len(PARAMETER_COMBINATIONS)} parameter combinations...\n")

    for fast_window, slow_window in PARAMETER_COMBINATIONS:
        print(
            f"Testing fast={fast_window}, "
            f"slow={slow_window}..."
        )

        backtest_results = run_backtest(
            prices,
            fast_window=fast_window,
            slow_window=slow_window,
            cost_bps=COST_BPS,
        )

        strategy_metrics = calculate_metrics(
            backtest_results["strategy_return"]
        )

        row = {
            "ticker": TICKER,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "cost_bps": COST_BPS,
        }

        # Add strategy performance metrics.
        for metric_name, value in strategy_metrics.items():
            row[f"strategy_{metric_name}"] = value

        # Add buy-and-hold benchmark metrics.
        for metric_name, value in benchmark_metrics.items():
            row[f"buy_hold_{metric_name}"] = value

        results.append(row)

    # Build a table and rank by strategy Sharpe ratio.
    results_df = pd.DataFrame(results)

    sharpe_column = "strategy_sharpe"

    if sharpe_column in results_df.columns:
        results_df = results_df.sort_values(
            by=sharpe_column,
            ascending=False,
        )

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Display the most useful columns.
    display_columns = [
        "fast_window",
        "slow_window",
        "strategy_total_return",
        "strategy_cagr",
        "strategy_annualized_volatility",
        "strategy_sharpe",
        "strategy_max_drawdown",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in results_df.columns
    ]

    print("\nParameter sensitivity results:")
    print(results_df[display_columns].to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
