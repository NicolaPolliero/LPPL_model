import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --- LPPLS Fit Visualization ---

def plot_lppls_results(price_series, results, title="LPPL Rolling Calibration — Critical Times"):
    """
    Plot observed log-price with vertical lines for predicted critical times (t_c).

    Parameters
    ----------
    price_series : pd.Series
        Series of observed prices with DateTime index.
    results : pd.DataFrame
        DataFrame of LPPLS fit results containing at least 'tc' and 'sign' columns.
    title : str, optional
        Title of the plot. Default is 'LPPL Rolling Calibration — Critical Times'.
    """
    if price_series.empty or results.empty:
        raise ValueError("Both 'price_series' and 'results' must be non-empty.")

    start_date = price_series.index.min()
    end_date = price_series.index.max()
    t0 = price_series.index[0]

    plt.figure(figsize=(12, 6))
    plt.plot(price_series.index, np.log(price_series.values),
             label="Observed log-price", lw=1.4, color="black")

    # --- Draw vertical t_c lines ---
    for _, row in results.iterrows():
        tc_years = row["tc"]
        tc_date = t0 + pd.to_timedelta(tc_years * 365.25, "D")
        color = "green" if row.get("sign", 1) > 0 else "red"
        plt.axvline(tc_date, color=color, alpha=0.5)

    plt.xlim(start_date, end_date)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("log(Price)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



def plot_tc_distribution(tc_predicted_from_start, symbol=None, color="royalblue"):
    """
    Plot the Kernel Density Estimate (KDE) of predicted critical times (t_c)
    relative to the start of the analyzed time series.

    Parameters
    ----------
    tc_predicted_from_start : array-like
        A 1D array or list of predicted critical times (in years since the
        start of the analyzed period). These values are typically obtained
        from LPPLS calibrations and represent predicted bubble burst times.
    symbol : str, optional
        Optional label (e.g., asset ticker symbol) used in the plot title.
        If None, a generic title will be shown.
    color : str, optional
        Color used to fill the KDE plot. Default is "royalblue".
    """
    plt.figure(figsize=(10, 5))
    sns.kdeplot(x=tc_predicted_from_start, fill=True, color=color, bw_adjust=0.3)
    plt.axvline(0, color="black", linestyle="--", lw=1, alpha=0.6)
    plt.xlabel("Predicted $t_c$ (years relative to target date)")
    if symbol:
        plt.title(f"KDE of Predicted $t_c$ — {symbol}")
    else:
        plt.title("KDE of Predicted $t_c$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_combined_tc_distributions(tc_predicted_from_start_1, tc_predicted_from_start_2, symbol=None):
    """
    Plot overlapping KDE distributions of predicted critical times (t_c)
    obtained from two different LPPLS calibration approaches, such as:
    - varying calibration dates
    - varying window sizes

    Parameters
    ----------
    tc_predicted_from_start_1 : array-like
        1D array or list of predicted t_c values (in years since start)
        obtained from the first calibration method (e.g., varying calibration dates).
    tc_predicted_from_start_2 : array-like
        1D array or list of predicted t_c values (in years since start)
        obtained from the second calibration method (e.g., varying window sizes).
    symbol : str, optional
        Optional label (e.g., asset ticker symbol) used in the plot title.
        If None, a generic title will be shown.
    """
    plt.figure(figsize=(10, 5))

    sns.kdeplot(
        x=tc_predicted_from_start_1, fill=True, color="darkorange",
        bw_adjust=0.4, alpha=0.6, label="Varying Calibration Dates"
    )
    sns.kdeplot(
        x=tc_predicted_from_start_2, fill=True, color="royalblue",
        bw_adjust=0.4, alpha=0.6, label="Varying Window Sizes"
    )

    plt.axvline(0, color="black", linestyle="--", lw=1, alpha=0.7)
    plt.xlabel("Predicted $t_c$ relative to target date (years)")
    if symbol:
        plt.title(f"KDE Comparison of Predicted $t_c$ — {symbol}")
    else:
        plt.title("KDE Comparison of Predicted $t_c$")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()