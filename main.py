
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_loader import load_prices
from backtest import run_backtest
from metrics import calculate_metrics
from parameter_heatmap import (
    COST_BPS,
    PERIOD,
    TICKER,
    build_heatmap,
    evaluation_window,
    select_parameters,
)


def print_metrics(name, returns):
    metrics = calculate_metrics(returns)

    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"  {metric:25s}: {value: .4f}")


def main():
    prices = load_prices(TICKER, PERIOD).dropna().sort_index()

    # The windows are chosen from the in-sample grid only, so the
    # out-of-sample period plays no part in picking them.
    print("Scoring the parameter grid on the in-sample period...")
    grid_results = build_heatmap(prices)
    fast_window, slow_window = select_parameters(grid_results)
    cost_bps = COST_BPS
    print(f"Selected fast={fast_window}, slow={slow_window}")

    results = run_backtest(
        prices,
        fast_window=fast_window,
        slow_window=slow_window,
        cost_bps=cost_bps,
    )

    # In-sample is the same rows the grid was scored on (after the longest
    # warm-up); out-of-sample is everything after it. Moving averages at the
    # start of out-of-sample may use earlier prices, which were known then.
    in_sample_window = evaluation_window(prices)
    split_index = in_sample_window.stop

    in_sample = results.iloc[in_sample_window]
    out_sample = results.iloc[split_index:]

    print(f"\nTotal observations: {len(results)}")
    print(
        f"In-sample observations: {len(in_sample)} "
        f"({in_sample.index[0].date()} to {in_sample.index[-1].date()})"
    )
    print(
        f"Out-of-sample observations: {len(out_sample)} "
        f"({out_sample.index[0].date()} to {out_sample.index[-1].date()})"
    )

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

    # Plot equity from the start of in-sample scoring, both rebased to $1.
    scored = results.iloc[in_sample_window.start:]
    equity = (1 + scored[["strategy_return", "buy_hold_return"]]).cumprod()
    equity.columns = ["strategy_equity", "buy_hold_equity"]

    ax = equity.plot(
        figsize=(11, 6),
        title=f"{TICKER}: Moving Average vs Buy-and-Hold",
        ylabel="Growth of $1",
    )
    ax.axvline(
        out_sample.index[0],
        color="gray",
        linestyle="--",
        label="Out-of-sample start",
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
