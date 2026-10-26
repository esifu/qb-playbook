
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_loader import load_prices
from backtest import run_backtest
from metrics import calculate_metrics


def print_metrics(name, returns):
    metrics = calculate_metrics(returns)

    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"  {metric:25s}: {value: .4f}")


def main():
    prices = load_prices("SPY", "10y")

    # Keep the strategy parameters fixed for this experiment.
    fast_window = 10
    slow_window = 100
    cost_bps = 6.0
    results = run_backtest(
        prices,
        fast_window=fast_window,
        slow_window=slow_window,
        cost_bps=cost_bps,
    )

    # Split chronologically: first 70% and final 30%.
    split_index = int(len(results) * 0.70)

    in_sample = results.iloc[:split_index]
    out_sample = results.iloc[split_index:]

    print(f"Total observations: {len(results)}")
    print(f"In-sample observations: {len(in_sample)}")
    print(f"Out-of-sample observations: {len(out_sample)}")

    for period_name, period in [
        ("IN-SAMPLE: Moving average", in_sample["strategy_return"]),
        ("IN-SAMPLE: Buy and hold", in_sample["buy_hold_return"]),
        ("OUT-OF-SAMPLE: Moving average", out_sample["strategy_return"]),
        ("OUT-OF-SAMPLE: Buy and hold", out_sample["buy_hold_return"]),
    ]:
        print_metrics(period_name, period)

    # Save the full backtest results.
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results.to_csv(output_dir / "backtest_results.csv")

    # Plot the full-period equity curves.
    ax = results[["strategy_equity", "buy_hold_equity"]].plot(
        figsize=(11, 6),
        title="SPY: Moving Average vs Buy-and-Hold",
        ylabel="Growth of $1",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color="none",
            label=f"Parameters: fast={fast_window}, slow={slow_window}, cost={cost_bps:g} bps",
        )
    )
    ax.legend(handles=handles, labels=labels + [handles[-1].get_label()])
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
