
import matplotlib.pyplot as plt

from data_loader import load_prices
from backtest import run_backtest
from metrics import calculate_metrics


def main():
    prices = load_prices("SPY", "10y")

    results = run_backtest(
        prices,
        fast_window=20,
        slow_window=100,
        cost_bps=5.0,
    )

    strategies = {
        "Moving average": results["strategy_return"],
        "Buy and hold": results["buy_hold_return"],
    }

    print("\nPerformance summary")
    print("=" * 65)

    for name, returns in strategies.items():
        metrics = calculate_metrics(returns)

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric:25s}: {value: .4f}")

    results[["strategy_equity", "buy_hold_equity"]].plot(
        figsize=(11, 6),
        title="SPY: Moving Average Strategy vs Buy-and-Hold",
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
