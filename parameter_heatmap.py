from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from data_loader import load_prices
from backtest import run_backtest
from metrics import calculate_metrics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TICKER = "SPY"
PERIOD = "10y"
COST_BPS = 5.0

# Only the first 70% of the data is used, so the final 30% stays
# untouched for a genuine out-of-sample test in main.py.
IN_SAMPLE_FRACTION = 0.70

FAST_WINDOWS = list(range(5, 55, 5))
SLOW_WINDOWS = list(range(50, 275, 25))

OUTPUT_DIR = Path("results")
OUTPUT_CSV = OUTPUT_DIR / "parameter_heatmap.csv"
OUTPUT_PNG = OUTPUT_DIR / "parameter_heatmap.png"

# Diverging colours: blue beats buy-and-hold, red trails it, gray is equal.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#e34948", "#f0efec", "#2a78d6"]
)


# --------------------------------------------------
# Grid evaluation
# --------------------------------------------------

def evaluation_window(prices: pd.Series) -> slice:
    """
    Rows every combination is scored on.

    Scoring starts after the longest slow window has warmed up, so no
    combination is penalised for days it could not trade, and stops at
    the end of the in-sample period.
    """
    start = max(SLOW_WINDOWS)
    end = int(len(prices) * IN_SAMPLE_FRACTION)

    if end <= start:
        raise ValueError("In-sample period is shorter than the slowest window")

    return slice(start, end)


def run_grid(prices: pd.Series, window: slice) -> pd.DataFrame:
    rows = []

    for fast_window in FAST_WINDOWS:
        for slow_window in SLOW_WINDOWS:
            if fast_window >= slow_window:
                continue

            backtest_results = run_backtest(
                prices,
                fast_window=fast_window,
                slow_window=slow_window,
                cost_bps=COST_BPS,
            ).iloc[window]

            metrics = calculate_metrics(backtest_results["strategy_return"])

            rows.append({
                "fast_window": fast_window,
                "slow_window": slow_window,
                **metrics,
            })

    return pd.DataFrame(rows)


def neighbourhood_mean(grid: pd.DataFrame) -> pd.DataFrame:
    """Average each cell with its valid neighbours (3x3 block)."""
    values = grid.to_numpy()
    smoothed = np.full_like(values, np.nan)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            block = values[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            smoothed[i, j] = np.nanmean(block)

    return pd.DataFrame(smoothed, index=grid.index, columns=grid.columns)


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_heatmap(ax, grid, benchmark_value, title, value_format):
    values = grid.to_numpy()
    spread = np.nanmax(np.abs(values - benchmark_value))
    norm = TwoSlopeNorm(
        vcenter=benchmark_value,
        vmin=benchmark_value - spread,
        vmax=benchmark_value + spread,
    )

    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=DIVERGING_CMAP,
        norm=norm,
        origin="lower",
        aspect="auto",
    )

    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("Slow window (days)")
    ax.set_ylabel("Fast window (days)")
    ax.set_title(title)

    # Annotate each cell; switch to white text on strongly coloured cells.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            strong = abs(value - benchmark_value) > 0.6 * spread
            ax.text(
                j, i, format(value, value_format),
                ha="center", va="center", fontsize=7,
                color="white" if strong else "#1a1a19",
            )

    # Outline the best cell.
    best_i, best_j = np.unravel_index(np.nanargmax(values), values.shape)
    ax.add_patch(plt.Rectangle(
        (best_j - 0.5, best_i - 0.5), 1, 1,
        fill=False, edgecolor="#1a1a19", linewidth=2,
    ))

    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label(f"Gray = buy-and-hold ({benchmark_value:{value_format}})")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"Loading historical prices for {TICKER}...")
    prices = load_prices(ticker=TICKER, period=PERIOD).dropna().sort_index()

    window = evaluation_window(prices)
    start_date = prices.index[window.start].date()
    end_date = prices.index[window.stop - 1].date()
    print(f"Scoring in-sample period {start_date} to {end_date}")

    results_df = run_grid(prices, window)

    # Buy-and-hold over exactly the same rows.
    benchmark = run_backtest(prices, cost_bps=COST_BPS).iloc[window]
    benchmark_metrics = calculate_metrics(benchmark["buy_hold_return"])

    sharpe_grid = results_df.pivot(
        index="fast_window", columns="slow_window", values="sharpe_ratio"
    )
    cagr_grid = results_df.pivot(
        index="fast_window", columns="slow_window", values="cagr"
    )
    smoothed_sharpe = neighbourhood_mean(sharpe_grid)

    results_df = results_df.merge(
        smoothed_sharpe.stack().rename("neighbourhood_sharpe").reset_index(),
        on=["fast_window", "slow_window"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    best_raw = results_df.loc[results_df["sharpe_ratio"].idxmax()]
    best_robust = results_df.loc[results_df["neighbourhood_sharpe"].idxmax()]

    print(f"\nBuy-and-hold Sharpe: {benchmark_metrics['sharpe_ratio']:.3f}")
    print(
        f"Best single cell:    fast={int(best_raw['fast_window'])}, "
        f"slow={int(best_raw['slow_window'])}, "
        f"Sharpe={best_raw['sharpe_ratio']:.3f}"
    )
    print(
        f"Most robust region:  fast={int(best_robust['fast_window'])}, "
        f"slow={int(best_robust['slow_window'])}, "
        f"neighbourhood Sharpe={best_robust['neighbourhood_sharpe']:.3f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    plot_heatmap(
        axes[0], sharpe_grid, benchmark_metrics["sharpe_ratio"],
        "Sharpe ratio", ".2f",
    )
    plot_heatmap(
        axes[1], cagr_grid, benchmark_metrics["cagr"],
        "CAGR", ".1%",
    )
    fig.suptitle(
        f"{TICKER} moving-average grid, in-sample {start_date} to {end_date}, "
        f"cost={COST_BPS:g} bps (outlined = best cell)"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)

    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Heatmap saved to: {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
