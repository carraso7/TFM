# -*- coding: utf-8 -*-
"""Fit Gumbel log-return-period curves and export plots and CSV summaries.

Reads daily multi-station streamflow from a semicolon-separated text file,
fits ``y = a * ln(T) + b`` to ranked annual maxima per station, and writes
per-station figures plus aggregated CSV outputs under ``output_data/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output_data"
STREAMFLOW_FILE = SCRIPT_DIR / "inputs_selected_stations_for_ret_period.txt"
STATIONS = ["018", "061", "062", "080", "170", "271"]
RETURN_PERIODS = [0.5, 1, 2, 5, 10]


def func(x, a, b):
    """Gumbel-style log return-period relationship ``y = a * ln(x) + b``."""
    return a * np.log(x) + b


def load_streamflow(path: Path) -> pd.DataFrame:
    """Load multi-station daily streamflow from a return-period export file.

    Input file structure (semicolon-separated):
        Row 0: ``Time;<station_ids...>``
        Row 1: ``station catchment;<catchment names...>``
        Row 2: ``station name;<station names...>``
        Remaining rows: ``YYYY-MM-DD;<streamflow values...>``

    Args:
        path: Path to the streamflow text file.

    Returns:
        DataFrame indexed by datetime with one float column per station id.
    """
    streamflow = pd.read_csv(path, sep=";")
    streamflow.drop([0, 1], axis=0, inplace=True)
    streamflow.set_index("Time", drop=True, inplace=True)
    streamflow.index = pd.to_datetime(streamflow.index, format="%Y-%m-%d")
    return streamflow


def fit_station_return_periods(
    station_id: str,
    streamflow: pd.DataFrame,
    return_periods: list[float],
) -> tuple[pd.DataFrame, np.ndarray, float, dict[float, float]]:
    """Rank annual maxima, fit a log return-period curve, and interpolate flows.

    Args:
        station_id: Column name in ``streamflow`` for the target station.
        streamflow: Daily streamflow DataFrame indexed by date.
        return_periods: Return periods in years for which to predict flow.

    Returns:
        Tuple of:
            - Daily streamflow column as a one-column DataFrame.
            - Ranked annual-maxima table with probabilities and fitted values.
            - Optimal parameters ``(a, b)`` for ``func``.
            - R² score of the log fit.
            - Dict mapping each requested return period to fitted streamflow (m³/s).
    """
    y = pd.DataFrame(streamflow[station_id]).astype(float)
    y_max_anual = y.resample("Y").max()
    y_sorted = y_max_anual.sort_values(by=y_max_anual.columns[0], ascending=False)
    y_sorted = y_sorted.reset_index()
    y_sorted["n_ocur"] = y_sorted.index + 1
    y_sorted["prob"] = y_sorted["n_ocur"] / (len(y_sorted) + 1)
    y_sorted["return period"] = 1 / y_sorted["prob"]

    popt, _pcov = curve_fit(func, y_sorted["return period"], y_sorted[station_id])
    y_sorted["y_pred"] = func(y_sorted["return period"], *popt)
    r2 = r2_score(y_sorted[station_id], y_sorted["y_pred"])

    streamflows_by_period = {
        period: float(func(period, *popt)) for period in return_periods
    }
    return y, y_sorted, popt, r2, streamflows_by_period


def save_return_period_fit_plot(
    station_id: str,
    y_sorted: pd.DataFrame,
    popt: np.ndarray,
    output_dir: Path,
) -> Path:
    """Save a return-period fit scatter plot with the fitted log curve.

    Args:
        station_id: Station identifier used in the plot title and filename.
        y_sorted: Ranked annual maxima with ``return period`` and observed flow.
        popt: Fitted ``(a, b)`` parameters for ``func``.
        output_dir: Directory where the PNG figure is written.

    Returns:
        Path to the saved PNG file.
    """
    figure_path = output_dir / f"station_{station_id}_return_period_fit.png"
    plt.figure()
    plt.plot(y_sorted["return period"], y_sorted[station_id], "bo", label="data")
    plt.plot(
        y_sorted["return period"],
        func(y_sorted["return period"], *popt),
        "-",
        label="fit",
    )
    plt.xlabel("Return period (years)")
    plt.ylabel("Annual maximum streamflow (m3/s)")
    plt.title(f"Station {station_id} - return period fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return figure_path


def save_hydrograph_plot(
    station_id: str,
    streamflow: pd.DataFrame,
    daily_streamflow: pd.DataFrame,
    streamflows_by_period: dict[float, float],
    return_periods: list[float],
    output_dir: Path,
) -> Path:
    """Plot observed daily hydrograph with horizontal return-period flow lines.

    Args:
        station_id: Station identifier for titles and filenames.
        streamflow: Full daily streamflow DataFrame indexed by date.
        daily_streamflow: One-column daily series for the target station.
        streamflows_by_period: Fitted flow values keyed by return period (years).
        return_periods: Return periods to annotate on the plot.
        output_dir: Directory where the PNG figure is written.

    Returns:
        Path to the saved PNG file.
    """
    figure_path = output_dir / f"station_{station_id}_hydrograph_return_periods.png"
    plt.figure()
    plt.plot(streamflow.index, daily_streamflow, "royalblue", label="observed data")
    colors = plt.cm.tab10(np.linspace(0, 1, len(return_periods)))
    label_x = streamflow.index[int(len(streamflow.index) * 0.98)]
    for index, period in enumerate(return_periods):
        flow_value = streamflows_by_period[period]
        plt.axhline(y=flow_value, color=colors[index], linestyle="--", linewidth=1)
        plt.text(
            label_x,
            flow_value + 0.1 * flow_value,
            f"$Q_{{T{period}}}$",
            fontsize=10,
            color="k",
        )
    plt.ylabel("Streamflow (m3/s)")
    plt.xlabel("Date")
    plt.ylim(0)
    plt.xlim(streamflow.index[0], streamflow.index[-1])
    plt.title(f"Gauging station {station_id}")
    plt.xticks(rotation=60)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return figure_path


def save_station_csvs(
    station_id: str,
    y_sorted: pd.DataFrame,
    popt: np.ndarray,
    r2: float,
    streamflows_by_period: dict[float, float],
    output_dir: Path,
) -> None:
    """Write per-station CSV files for ranked maxima, fit params, and flows.

    Output files (under ``output_dir``):
        - ``station_<id>_annual_maxima_ranked.csv``
        - ``station_<id>_fitted_parameters.csv``
        - ``station_<id>_return_period_streamflows.csv``

    Args:
        station_id: Target station identifier.
        y_sorted: Ranked annual maxima with probabilities and predictions.
        popt: Fitted log-curve parameters ``(a, b)``.
        r2: Coefficient of determination for the fit.
        streamflows_by_period: Interpolated flows by return period (years).
        output_dir: Destination folder for CSV exports.
    """
    annual_maxima_path = output_dir / f"station_{station_id}_annual_maxima_ranked.csv"
    y_sorted.to_csv(annual_maxima_path, index=False)

    fitted_parameters = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "log_coefficient_a": popt[0],
                "log_coefficient_b": popt[1],
                "r2_score": r2,
                "equation": f"{popt[0]}*ln(x)+{popt[1]}",
            }
        ]
    )
    fitted_parameters.to_csv(
        output_dir / f"station_{station_id}_fitted_parameters.csv",
        index=False,
    )

    streamflows = pd.DataFrame(
        [
            {
                "station_id": station_id,
                "return_period_years": period,
                "streamflow_m3s": flow,
            }
            for period, flow in streamflows_by_period.items()
        ]
    )
    streamflows.to_csv(
        output_dir / f"station_{station_id}_return_period_streamflows.csv",
        index=False,
    )


def save_summary_csvs(
    fitted_rows: list[dict[str, object]],
    streamflow_rows: list[dict[str, object]],
    return_periods: list[float],
    output_dir: Path,
) -> None:
    """Write combined CSV summaries across all processed stations.

    Output files (under ``output_dir``):
        - ``all_stations_fitted_parameters.csv``
        - ``all_stations_return_period_streamflows.csv`` (long format)
        - ``all_stations_return_period_streamflows_wide.csv`` (wide format)

    Args:
        fitted_rows: One dict per station with log-fit metadata.
        streamflow_rows: Long-format rows with station, period, and flow.
        return_periods: Return periods used as wide-format column suffixes.
        output_dir: Destination folder for summary CSV files.
    """
    fitted_df = pd.DataFrame(fitted_rows)
    fitted_df.to_csv(output_dir / "all_stations_fitted_parameters.csv", index=False)

    streamflows_long = pd.DataFrame(streamflow_rows)
    streamflows_long.to_csv(
        output_dir / "all_stations_return_period_streamflows.csv",
        index=False,
    )

    wide_rows: list[dict[str, object]] = []
    for station_id in fitted_df["station_id"]:
        station_flows = {
            row["return_period_years"]: row["streamflow_m3s"]
            for row in streamflow_rows
            if row["station_id"] == station_id
        }
        row = {"station_id": station_id}
        for period in return_periods:
            row[f"T{period}_streamflow_m3s"] = station_flows.get(period)
        wide_rows.append(row)

    pd.DataFrame(wide_rows).to_csv(
        output_dir / "all_stations_return_period_streamflows_wide.csv",
        index=False,
    )


def main() -> None:
    """Run return-period analysis for all configured stations and save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    streamflow = load_streamflow(STREAMFLOW_FILE)

    fitted_rows: list[dict[str, object]] = []
    streamflow_rows: list[dict[str, object]] = []

    for station_id in STATIONS:
        daily_streamflow, y_sorted, popt, r2, streamflows_by_period = (
            fit_station_return_periods(station_id, streamflow, RETURN_PERIODS)
        )

        save_return_period_fit_plot(station_id, y_sorted, popt, OUTPUT_DIR)
        save_hydrograph_plot(
            station_id,
            streamflow,
            daily_streamflow,
            streamflows_by_period,
            RETURN_PERIODS,
            OUTPUT_DIR,
        )
        save_station_csvs(
            station_id,
            y_sorted,
            popt,
            r2,
            streamflows_by_period,
            OUTPUT_DIR,
        )

        fitted_rows.append(
            {
                "station_id": station_id,
                "log_coefficient_a": popt[0],
                "log_coefficient_b": popt[1],
                "r2_score": r2,
                "equation": f"{popt[0]}*ln(x)+{popt[1]}",
            }
        )
        for period, flow in streamflows_by_period.items():
            streamflow_rows.append(
                {
                    "station_id": station_id,
                    "return_period_years": period,
                    "streamflow_m3s": flow,
                }
            )

        print(f"For station {station_id}:")
        print(f"  Logarithm function: {popt[0]}*ln(x)+{popt[1]}")
        print(f"  R2 score: {r2:.4f}")
        print(f"  Return period streamflows: {streamflows_by_period}")

    save_summary_csvs(fitted_rows, streamflow_rows, RETURN_PERIODS, OUTPUT_DIR)
    print(f"\nSaved figures and CSV files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
