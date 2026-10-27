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

#(fast_window, slow_window).
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

    # Calculate buy-and-hold benchmark metrics once.
    baseline = run_backtest(
        prices,
        fast_window=20,
        slow_window=100,
        cost_bps=COST_BPS,
    )

    benchmark_metrics = calculate_metrics(
        baseline["buy_hold_return"]
    )

    print(
        f"\nTesting {len(PARAMETER_COMBINATIONS)} "
        "parameter combinations...\n"
    )

    # Run each moving-average combination.
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

        # Store strategy metrics.
        for metric_name, value in strategy_metrics.items():
            row[f"strategy_{metric_name}"] = value

        # Store buy-and-hold benchmark metrics.
        for metric_name, value in benchmark_metrics.items():
            row[f"buy_hold_{metric_name}"] = value

        results.append(row)

    # Convert results into a DataFrame.
    results_df = pd.DataFrame(results)

    # Expected metric names from metrics.py.
    cagr_column = "strategy_cagr"
    sharpe_column = "strategy_sharpe_ratio"
    volatility_column = "strategy_annualized_volatility"
    drawdown_column = "strategy_max_drawdown"

    # Check that required columns exist.
    print("\nAvailable columns:", results_df.columns.tolist())

    required_columns = [
        cagr_column,
        sharpe_column,
        volatility_column,
        drawdown_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Expected metric columns were not found: "
            f"{missing_columns}. "
            "Check the keys returned by calculate_metrics() "
            "in metrics.py and update the column names above."
        )

    # Sort results by Sharpe ratio, highest first.
    results_df = results_df.sort_values(
        by=sharpe_column,
        ascending=False,
    )

    # Save results to CSV.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Display comparison table.
    display_columns = [
        "fast_window",
        "slow_window",
        "strategy_total_return",
        cagr_column,
        volatility_column,
        sharpe_column,
        drawdown_column,
    ]

    display_columns = [
        column
        for column in display_columns
        if column in results_df.columns
    ]

    print("\nParameter sensitivity results:")
    print(results_df[display_columns].to_string(index=False))

    # --------------------------------------------------
    # Best risk-adjusted performance
    # --------------------------------------------------

    best_sharpe = results_df.loc[
        results_df[sharpe_column].idxmax()
    ]

    print("\n" + "=" * 50)
    print("BEST RISK-ADJUSTED PERFORMANCE")
    print("=" * 50)
    print("Ranking metric: Sharpe ratio")
    print(f"Fast window: {int(best_sharpe['fast_window'])} days")
    print(f"Slow window: {int(best_sharpe['slow_window'])} days")
    print(f"CAGR: {best_sharpe[cagr_column]:.2%}")
    print(f"Sharpe ratio: {best_sharpe[sharpe_column]:.3f}")
    print(
        f"Annualized volatility: "
        f"{best_sharpe[volatility_column]:.2%}"
    )
    print(
        f"Maximum drawdown: "
        f"{best_sharpe[drawdown_column]:.2%}"
    )

    # --------------------------------------------------
    # Best pure return
    # --------------------------------------------------

    best_return = results_df.loc[
        results_df[cagr_column].idxmax()
    ]

    print("\n" + "=" * 50)
    print("BEST PURE RETURN")
    print("=" * 50)
    print("Ranking metric: CAGR")
    print(f"Fast window: {int(best_return['fast_window'])} days")
    print(f"Slow window: {int(best_return['slow_window'])} days")
    print(f"CAGR: {best_return[cagr_column]:.2%}")
    print(f"Sharpe ratio: {best_return[sharpe_column]:.3f}")
    print(
        f"Annualized volatility: "
        f"{best_return[volatility_column]:.2%}"
    )
    print(
        f"Maximum drawdown: "
        f"{best_return[drawdown_column]:.2%}"
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
