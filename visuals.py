#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from bokeh.models import (
    Arrow,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LabelSet,
    Legend,
    LegendItem,
    LinearColorMapper,
    NormalHead,
    Range1d,
    WMTSTileSource,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, show

from adj_matrix_visualize_maps_GNNs import export_graph_for_qgis


OBSERVED_LINE_COLOR = "#000000"
PREDICTED_LINE_COLOR = "#2b9f9f"
PREDICTION_LINE_WIDTH = 1.25
FONTSIZE_DEFAULT = 15

WEIGHTED_ADJ_BOXPLOT_LABELS = {
    "dense": "comp.",
    "all_paths": "dist.",
    "hydro": "clos. dist.",
    "river_dist": "riv. dist.",
}

ERROR_METRICS = ("MSE", "NSE", "RMSE", "nRMSE", "KGE")
ErrorMetric = Literal["MSE", "NSE", "RMSE", "nRMSE", "KGE"]
ConchiModel = Literal["LSTM", "MC-LSTM"]
ConchiScenario = Literal["TS1", "TS2", "TS3"]

DEFAULT_CONCHI_NSE_PATH = (
    Path(__file__).resolve().parent.parent / "results_conchi" / "ESTACIONES_AFORO_NSE.csv"
) # Also in "D:\streamflow_prediction\results_conchi\ESTACIONES_AFORO_NSE.csv"
DEFAULT_RETURN_PERIODS_PATH = "/mnt/d/streamflow_prediction/summary_return_periods.csv"
DEFAULT_CONCHI_RETURN_PERIOD_NRMSE_PATH = (
    "/mnt/d/streamflow_prediction/results_conchi/selected_conchi_results_ret_periods.csv"
)
MISSING_CONCHI_COLOR = "#9ca3af"

RETURN_PERIOD_LABELS = {
    0.5: "T0.5",
    1: "T1",
    1.0: "T1",
    2: "T2",
    2.0: "T2",
    5: "T5",
    5.0: "T5",
    10: "T10",
    10.0: "T10",
}
RETURN_PERIOD_VALUE_TO_COLUMN = {
    0.5: "T0.5",
    1.0: "T1",
    2.0: "T2",
    5.0: "T5",
    10.0: "T10",
}
DEFAULT_RETURN_PERIODS = [0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_RETURN_PERIOD_LINEPLOT_GRID_ROWS = 2
DEFAULT_RETURN_PERIOD_LINEPLOT_GRID_COLS = 2
DEFAULT_RETURN_PERIOD_LINEPLOT_SHOW_ONLY_STATION_ID = True
DEFAULT_RETURN_PERIOD_LINEPLOT_INDIVIDUAL_DIRNAME = "lineplot_return_periods"
ConchiTrainingScenario = Literal["C1", "C2", "C3", "TS1"]
RETURN_PERIOD_MODEL_COLORS = {
    "GNN-LSTM": "#1f8a8a",
    "LSTM": "#e66101",
    "MC-LSTM": "#5e3c99",
}
KGE_COMPONENT_COLORS = {
    "r": "#2ca25f",
    "alpha": "#e66101",
    "beta": "#5e3c99",
}
KGE_PIE_FALLBACK_COLOR = "#9ca3af"
KGE_PIE_RADIUS = 18
KGE_MAP_STATION_LABEL_Y_OFFSET = 26
KGE_MAP_COMPONENT_LABEL_Y_OFFSET = -34
KGE_MAP_VIEW_PADDING_FRACTION = 0.26
MAP_SEPARATED_LEGEND_DEFAULT = True


def _validate_error_metric(error_metric: str) -> ErrorMetric:
    canonical_metrics = {
        "mse": "MSE",
        "nse": "NSE",
        "rmse": "RMSE",
        "nrmse": "nRMSE",
        "kge": "KGE",
    }
    metric = canonical_metrics.get(error_metric.strip().lower())
    if metric is None:
        raise ValueError(f"error_metric must be one of {ERROR_METRICS}, got {error_metric!r}")
    return metric


def compute_mse(observed: np.ndarray | list[float], predicted: np.ndarray | list[float]) -> float:
    observed_arr, predicted_arr = _paired_finite_arrays(observed, predicted)
    if observed_arr.size == 0:
        return float("nan")
    return float(np.mean((observed_arr - predicted_arr) ** 2))


def compute_nse(observed: np.ndarray | list[float], predicted: np.ndarray | list[float]) -> float:
    observed_arr, predicted_arr = _paired_finite_arrays(observed, predicted)
    if observed_arr.size == 0:
        return float("nan")
    denominator = float(np.sum((observed_arr - observed_arr.mean()) ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum((observed_arr - predicted_arr) ** 2) / denominator)


def compute_rmse(observed: np.ndarray | list[float], predicted: np.ndarray | list[float]) -> float:
    observed_arr, predicted_arr = _paired_finite_arrays(observed, predicted)
    if observed_arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((observed_arr - predicted_arr) ** 2)))


def compute_nrmse(observed: np.ndarray | list[float], predicted: np.ndarray | list[float]) -> float:
    observed_arr, predicted_arr = _paired_finite_arrays(observed, predicted)
    if observed_arr.size == 0:
        return float("nan")
    mean_observed = float(observed_arr.mean())
    if mean_observed == 0.0:
        return float("nan")
    return compute_rmse(observed_arr, predicted_arr) / mean_observed


def compute_kge_components(
    observed: np.ndarray | list[float],
    predicted: np.ndarray | list[float],
) -> tuple[float, float, float]:
    """
    Return Pearson r, alpha (sigma_s / sigma_o), and beta (mu_s / mu_o) for KGE.
    """
    observed_arr, predicted_arr = _paired_finite_arrays(observed, predicted)
    if observed_arr.size < 2:
        return float("nan"), float("nan"), float("nan")

    mean_observed = float(observed_arr.mean())
    std_observed = float(observed_arr.std())
    if mean_observed == 0.0 or std_observed == 0.0:
        return float("nan"), float("nan"), float("nan")

    correlation = np.corrcoef(observed_arr, predicted_arr)[0, 1]
    if not np.isfinite(correlation):
        return float("nan"), float("nan"), float("nan")

    alpha = float(predicted_arr.std()) / std_observed
    beta = float(predicted_arr.mean()) / mean_observed
    return float(correlation), alpha, beta


def compute_kge(observed: np.ndarray | list[float], predicted: np.ndarray | list[float]) -> float:
    """
    Kling-Gupta efficiency (KGE).

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)

    where r is the Pearson correlation, alpha = sigma_s / sigma_o,
    and beta = mu_s / mu_o.
    """
    r, alpha, beta = compute_kge_components(observed, predicted)
    if not all(np.isfinite(value) for value in (r, alpha, beta)):
        return float("nan")
    return float(
        1.0
        - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    )


def _paired_finite_arrays(
    observed: np.ndarray | list[float],
    predicted: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    observed_arr = np.asarray(observed, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    if observed_arr.shape != predicted_arr.shape:
        raise ValueError("observed and predicted arrays must have the same shape")
    mask = np.isfinite(observed_arr) & np.isfinite(predicted_arr)
    return observed_arr[mask], predicted_arr[mask]


def compute_station_error(
    observed: np.ndarray | list[float],
    predicted: np.ndarray | list[float],
    error_metric: str = "NSE",
) -> float:
    metric = _validate_error_metric(error_metric)
    if metric == "MSE":
        return compute_mse(observed, predicted)
    if metric == "NSE":
        return compute_nse(observed, predicted)
    if metric == "RMSE":
        return compute_rmse(observed, predicted)
    if metric == "nRMSE":
        return compute_nrmse(observed, predicted)
    return compute_kge(observed, predicted)


def compute_error_by_station_from_frames(
    station_frames: dict[str, pd.DataFrame],
    error_metric: str = "NSE",
    *,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> dict[str, float]:
    metric = _validate_error_metric(error_metric)
    return {
        station_id: compute_station_error(
            frame[observed_col].to_numpy(dtype=np.float64),
            frame[predicted_col].to_numpy(dtype=np.float64),
            metric,
        )
        for station_id, frame in station_frames.items()
    }


def compute_kge_components_by_station_from_frames(
    station_frames: dict[str, pd.DataFrame],
    *,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> dict[str, tuple[float, float, float]]:
    return {
        station_id: compute_kge_components(
            frame[observed_col].to_numpy(dtype=np.float64),
            frame[predicted_col].to_numpy(dtype=np.float64),
        )
        for station_id, frame in station_frames.items()
    }


def plot_random_year_predictions(
    station_frames: dict[str, pd.DataFrame],
    num_stations: int = 4,
    visuals_dir: str | Path | None = None,
    seed: int = 42,
    filename_prefix: str = "prediction",
) -> None:
    if not station_frames:
        print("No stations available for plotting.")
        return

    rng = random.Random(seed)
    station_ids = list(station_frames.keys())
    selected_ids = rng.sample(station_ids, k=min(num_stations, len(station_ids)))

    if visuals_dir is not None:
        visuals_dir = Path(visuals_dir)
        visuals_dir.mkdir(parents=True, exist_ok=True)

    for station_id in selected_ids:
        data = station_frames[station_id]
        if data.empty:
            continue

        dates = pd.to_datetime(data["date"]).to_numpy()
        targets = data["observed"].to_numpy(dtype=np.float32)
        preds = data["predicted"].to_numpy(dtype=np.float32)

        years = pd.Series(dates).dt.year.to_numpy()
        station_years = sorted(set(years))
        if not station_years:
            continue
        year = rng.choice(station_years)
        year_mask = years == year
        if not year_mask.any():
            continue

        year_dates = np.asarray(dates)[year_mask]
        year_targets = targets[year_mask]
        year_preds = preds[year_mask]
        order = np.argsort(year_dates)
        year_dates = year_dates[order]
        year_targets = year_targets[order]
        year_preds = year_preds[order]

        plt.figure(figsize=(10, 4))
        plt.plot(year_dates, year_targets, label="Observed data", linewidth=2)
        plt.plot(year_dates, year_preds, label="Predicted data", linewidth=2)
        plt.title(f"Station {station_id} - {year}")
        plt.xlabel("Date")
        plt.ylabel("Streamflow")
        plt.legend()
        plt.tight_layout()
        if visuals_dir is not None:
            png_path = Path(visuals_dir) / f"{filename_prefix}_{station_id}_{year}.png"
            plt.savefig(png_path, dpi=150)
        plt.show()


def load_station_name_map(static_info_path: str | Path) -> dict[str, str]:
    static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
    if "station_id" not in static_info_df.columns:
        static_info_df = static_info_df.reset_index()
    if "Station name" not in static_info_df.columns:
        raise ValueError("static_info must include a 'Station name' column")

    station_names: dict[str, str] = {}
    for _, row in static_info_df.iterrows():
        station_id = row.get("station_id")
        name = row.get("Station name")
        if pd.isna(station_id) or pd.isna(name):
            continue
        station_names[str(station_id)] = str(name).strip()
    return station_names





def plot_test_years_predictions( ### TODO REVISAR VALORES SIMILARES 62 Y 80
    station_frames: dict[str, pd.DataFrame],
    test_years: list[int],
    station_ids: list[str] | None = None,
    station_names: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
    filename_prefix: str = "prediction",
    one_line: bool = True,
    year_range: tuple[int | None, int | None] = (None, None),
    test_start_date: pd.Timestamp | None = None,
    test_end_date: pd.Timestamp | None = None,
    show_month_labels: bool = True,
    show_plot: bool = False,
    separated_legend: bool = True,
    fontsize: float = FONTSIZE_DEFAULT,
) -> None:
    def _format_station_label(station_id: str, station_names: dict[str, str] | None) -> str:
        if not station_names:
            return station_id
        name = station_names.get(station_id)
        if not name:
            return station_id
        return f"{station_id}: {str(name).strip().replace(' en ', ' in ')}"

    def _apply_axis_font_size(axis: plt.Axes) -> None:
        axis.xaxis.label.set_size(fontsize)
        axis.yaxis.label.set_size(fontsize)
        axis.tick_params(labelsize=fontsize)
        title = axis.get_title()
        if title:
            axis.set_title(title, fontsize=fontsize)

    def _save_separated_legend(
        handles: list,
        labels: list[str],
        legend_path: Path,
    ) -> None:
        legend_fig, legend_axis = plt.subplots(figsize=(4, 0.5))
        legend_axis.axis("off")
        legend_fig.legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            fontsize=fontsize,
            ncol=len(labels),
        )
        legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight")
        plt.close(legend_fig)

    def _attach_legend(
        axis: plt.Axes,
        *,
        legend_path: Path | None = None,
    ) -> None:
        handles, labels = axis.get_legend_handles_labels()
        if not handles:
            return
        if separated_legend:
            if legend_path is not None:
                _save_separated_legend(handles, labels, legend_path)
            return
        axis.legend(handles, labels, loc="upper right", fontsize=fontsize)
    def _filter_years_by_range(
        years: list[int],
        year_range: tuple[int | None, int | None],
    ) -> list[int]:
        min_year, max_year = year_range
        filtered = years
        if min_year is not None:
            filtered = [year for year in filtered if year >= min_year]
        if max_year is not None:
            filtered = [year for year in filtered if year <= max_year]
        return filtered
    def _set_xlim_to_dates(axis: plt.Axes, dates: np.ndarray) -> None:
        if len(dates) == 0:
            return
        axis.set_xlim(pd.Timestamp(min(dates)), pd.Timestamp(max(dates)))
        axis.margins(x=0)

    def _apply_date_axis_labels(axis: plt.Axes) -> None:
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    def _build_test_plot_mask(
        dates: pd.Series,
        years: np.ndarray,
        years_with_data: list[int],
    ) -> np.ndarray:
        mask = np.isin(years, years_with_data)
        if test_start_date is not None:
            mask &= dates >= test_start_date
        if test_end_date is not None:
            mask &= dates <= test_end_date
        return mask

    def _plot_observed_predicted(
        axis: plt.Axes,
        dates: np.ndarray,
        targets: np.ndarray,
        preds: np.ndarray,
    ) -> None:
        axis.plot(
            dates,
            targets,
            label="Observed data",
            color=OBSERVED_LINE_COLOR,
            linewidth=PREDICTION_LINE_WIDTH,
        )
        axis.plot(
            dates,
            preds,
            label="Predicted data",
            color=PREDICTED_LINE_COLOR,
            linewidth=PREDICTION_LINE_WIDTH,
        )


    if not station_frames:
        print("No stations available for plotting.")
        return
    if not test_years:
        print("No test years provided for plotting.")
        return

    test_years = sorted(set(_filter_years_by_range(test_years, year_range)))
    if not test_years:
        print("No test years remain after applying year_range.")
        return

    available_ids = list(station_frames.keys())
    if station_ids is None:
        selected_ids = available_ids
    else:
        selected_ids = [station_id for station_id in station_ids if station_id in station_frames]
        missing_ids = [station_id for station_id in station_ids if station_id not in station_frames]
        for station_id in missing_ids:
            print(f"Station {station_id} not found in prediction frames; skipping.")

    if not selected_ids:
        print("No matching stations available for plotting.")
        return

    years_with_data = list(test_years)
    for station_id in selected_ids:
        data = station_frames[station_id]
        if data.empty:
            continue
        station_years = pd.to_datetime(data["date"]).dt.year.to_numpy()
        years_with_data = [year for year in test_years if np.any(station_years == year)]
        break

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    plot_date_starts: list[pd.Timestamp] = []
    plot_date_ends: list[pd.Timestamp] = []
    for station_id in selected_ids:
        data = station_frames[station_id]
        if data.empty:
            continue
        dates = pd.to_datetime(data["date"])
        years = dates.dt.year.to_numpy()
        plot_mask = _build_test_plot_mask(dates, years, years_with_data)
        if not plot_mask.any():
            continue
        plot_dates = dates[plot_mask]
        plot_date_starts.append(plot_dates.min())
        plot_date_ends.append(plot_dates.max())

    if plot_date_starts:
        day_range_start = min(plot_date_starts)
        day_range_end = max(plot_date_ends)
        num_days = (day_range_end - day_range_start).days + 1
        print(
            "Test set / plot_test_years_predictions day range: "
            f"{day_range_start.date()} to {day_range_end.date()} ({num_days} days)"
        )
    else:
        print("No test-set days available for plotting after applying filters.")
        return

    for station_id in selected_ids:
        data = station_frames[station_id]
        if data.empty:
            continue

        dates = pd.to_datetime(data["date"])
        targets = data["observed"].to_numpy(dtype=np.float32)
        preds = data["predicted"].to_numpy(dtype=np.float32)
        years = dates.dt.year.to_numpy()

        station_label = _format_station_label(station_id, station_names)
        test_mask = _build_test_plot_mask(dates, years, years_with_data)
        if not test_mask.any():
            continue
        test_dates = dates[test_mask].to_numpy()
        test_targets = targets[test_mask]
        test_preds = preds[test_mask]
        order = np.argsort(test_dates)
        test_dates = test_dates[order]
        test_targets = test_targets[order]
        test_preds = test_preds[order]

        if one_line:
            fig, axis = plt.subplots(1, 1, figsize=(12, 4))
            _plot_observed_predicted(axis, test_dates, test_targets, test_preds)
            axis.set_title(station_label)
            axis.set_ylabel("Streamflow [m³/s]")
            _set_xlim_to_dates(axis, test_dates)
            _apply_date_axis_labels(axis)
            _apply_axis_font_size(axis)
            legend_path = (
                output_dir / f"{filename_prefix}_{station_id}_legend.png"
                if output_dir is not None and separated_legend
                else None
            )
            _attach_legend(axis, legend_path=legend_path)
            fig.tight_layout()
        else:
            fig, axes = plt.subplots(
                len(years_with_data),
                1,
                figsize=(10, 3.5 * len(years_with_data)),
                sharex=False,
            )
            if len(years_with_data) == 1:
                axes = [axes]

            for axis, year in zip(axes, years_with_data):
                year_mask = (years == year) & test_mask
                if not year_mask.any():
                    continue
                year_dates = dates[year_mask].to_numpy()
                year_targets = targets[year_mask]
                year_preds = preds[year_mask]
                year_order = np.argsort(year_dates)
                year_dates = year_dates[year_order]
                year_targets = year_targets[year_order]
                year_preds = year_preds[year_order]

                _plot_observed_predicted(axis, year_dates, year_targets, year_preds)
                axis.set_title(f"{year}")
                axis.set_ylabel("Streamflow [m³/s]")
                _set_xlim_to_dates(axis, year_dates)
                _apply_date_axis_labels(axis)
                _apply_axis_font_size(axis)

            legend_path = (
                output_dir / f"{filename_prefix}_{station_id}_legend.png"
                if output_dir is not None and separated_legend
                else None
            )
            _attach_legend(axes[0], legend_path=legend_path)
            fig.tight_layout()

        if output_dir is not None:
            png_path = output_dir / f"{filename_prefix}_{station_id}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def _normalize_station_id_for_conchi(station_id: str) -> str: ### TODO REVISAR MÉTODO ENTERO
    normalized = str(station_id).strip().upper()
    if normalized.startswith("A"):
        normalized = normalized[1:]
    return normalized.zfill(3)


def _load_conchi_nse_by_station( ### TODO REVISAR MÉTODO ENTERO
    conchi_nse_path: str | Path,
    conchi_model: ConchiModel,
    conchi_scenario: ConchiScenario,
) -> dict[str, float]:
    path = Path(conchi_nse_path)
    with path.open(encoding="utf-8") as handle:
        scenario_row = handle.readline().strip().split(";")
        model_row = handle.readline().strip().split(";")

    column_idx: int | None = None
    for idx in range(2, len(scenario_row)):
        if scenario_row[idx] == conchi_scenario and model_row[idx] == conchi_model:
            column_idx = idx
            break
    if column_idx is None:
        raise ValueError(
            f"Could not find NSE column for scenario={conchi_scenario!r} "
            f"and model={conchi_model!r} in {path}"
        )

    conchi_df = pd.read_csv(path, sep=";", skiprows=2, dtype={"ID": str})
    if "ID" not in conchi_df.columns:
        raise ValueError(f"Conchi NSE file must include an ID column: {path}")

    nse_column = conchi_df.columns[column_idx]
    conchi_nse_by_station: dict[str, float] = {}
    for _, row in conchi_df.iterrows():
        station_id = row.get("ID")
        if pd.isna(station_id):
            continue
        value = row.get(nse_column)
        if pd.isna(value):
            continue
        conchi_nse_by_station[_normalize_station_id_for_conchi(str(station_id))] = float(value)
    return conchi_nse_by_station


def _red_white_blue_palette(num_colors: int = 256) -> list[str]: ### TODO REVISAR MÉTODO ENTERO
    red = np.array([215.0, 48.0, 39.0])
    white = np.array([255.0, 255.0, 255.0])
    blue = np.array([69.0, 117.0, 180.0])
    palette: list[str] = []
    for step in range(num_colors):
        fraction = step / (num_colors - 1)
        if fraction <= 0.5:
            blend = red * (1.0 - 2.0 * fraction) + white * (2.0 * fraction)
        else:
            blend = white * (2.0 - 2.0 * fraction) + blue * (2.0 * fraction - 1.0)
        palette.append(
            f"#{int(round(blend[0])):02x}{int(round(blend[1])):02x}{int(round(blend[2])):02x}"
        )
    return palette


def _symmetric_diverging_bounds(values: list[float], *, minimum_span: float = 0.05) -> tuple[float, float]: ### TODO REVISAR MÉTODO ENTERO
    finite_values = np.array([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite_values.size == 0:
        half_span = minimum_span
    else:
        half_span = max(float(np.max(np.abs(finite_values))), minimum_span)
    return -half_span, half_span


def _format_signed_difference(value: float | None) -> str: ### TODO REVISAR MÉTODO ENTERO
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:+.3f}"


def _lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    r_major = 6378137.0
    x = math.radians(lon) * r_major
    lat = max(min(lat, 89.9999), -89.9999)
    y = r_major * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _format_error_label(metric: str, value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return f"{metric}\nN/A"
    return f"{metric}\n{value:.2f}"


def _format_kge_components_label(r: float, alpha: float, beta: float) -> str:
    if not all(np.isfinite(value) for value in (r, alpha, beta)):
        return "r=N/A\nα=N/A\nβ=N/A"
    return f"r={r:.2f}\nα={alpha:.2f}\nβ={beta:.2f}"


def _compute_kge_component_shares(
    r: float,
    alpha: float,
    beta: float,
) -> tuple[float, float, float] | None:
    if not all(np.isfinite(value) for value in (r, alpha, beta)):
        return None
    r_term = (r - 1.0) ** 2
    alpha_term = (alpha - 1.0) ** 2
    beta_term = (beta - 1.0) ** 2
    total = r_term + alpha_term + beta_term
    if not np.isfinite(total) or total <= 0.0:
        return None
    return (
        float(r_term / total),
        float(alpha_term / total),
        float(beta_term / total),
    )


def _build_kge_pie_wedge_source(
    station_ids: list[str],
    coord_lookup: dict[str, tuple[float, float]],
    kge_components_by_station: dict[str, tuple[float, float, float]],
) -> tuple[ColumnDataSource, ColumnDataSource]:
    wedge_x: list[float] = []
    wedge_y: list[float] = []
    wedge_start_angle: list[float] = []
    wedge_end_angle: list[float] = []
    wedge_colors: list[str] = []
    wedge_station_ids: list[str] = []

    gray_x: list[float] = []
    gray_y: list[float] = []
    gray_station_ids: list[str] = []

    two_pi = 2.0 * math.pi
    component_order = ("r", "alpha", "beta")

    for station_id in station_ids:
        x_value, y_value = coord_lookup[station_id]
        components = kge_components_by_station.get(station_id, (float("nan"), float("nan"), float("nan")))
        shares = _compute_kge_component_shares(*components)
        if shares is None:
            gray_x.append(x_value)
            gray_y.append(y_value)
            gray_station_ids.append(station_id)
            continue

        angle = -math.pi / 2.0
        for component_name, share in zip(component_order, shares):
            if share <= 0.0:
                continue
            wedge_x.append(x_value)
            wedge_y.append(y_value)
            wedge_start_angle.append(angle)
            wedge_end_angle.append(angle + share * two_pi)
            wedge_colors.append(KGE_COMPONENT_COLORS[component_name])
            wedge_station_ids.append(station_id)
            angle += share * two_pi

    wedge_source = ColumnDataSource(
        {
            "x": wedge_x,
            "y": wedge_y,
            "start_angle": wedge_start_angle,
            "end_angle": wedge_end_angle,
            "color": wedge_colors,
            "station_id": wedge_station_ids,
        }
    )
    gray_source = ColumnDataSource(
        {
            "x": gray_x,
            "y": gray_y,
            "station_id": gray_station_ids,
        }
    )
    return wedge_source, gray_source


def _apply_map_view_padding(
    plot,
    x_coords: list[float],
    y_coords: list[float],
    *,
    padding_fraction: float = 0.12,
) -> None:
    if not x_coords or not y_coords:
        return
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = x_span * padding_fraction if x_span > 0 else 50_000.0
    y_pad = y_span * padding_fraction if y_span > 0 else 50_000.0
    plot.x_range = Range1d(x_min - x_pad, x_max + x_pad)
    plot.y_range = Range1d(y_min - y_pad, y_max + y_pad)


def _format_legend_interval_slug(low: float, high: float) -> str:
    def _fmt(value: float) -> str:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text if text else "0"

    return f"{_fmt(low)}_{_fmt(high)}"


def _map_color_bar_legend_png_path(
    output_png: str | Path,
    low: float,
    high: float,
) -> Path:
    output_png = Path(output_png)
    interval_slug = _format_legend_interval_slug(low, high)
    return output_png.parent / f"{output_png.stem}_legend_{interval_slug}.png"


def _map_categorical_legend_png_path(output_png: str | Path) -> Path:
    output_png = Path(output_png)
    return output_png.parent / f"{output_png.stem}_legend.png"


def _export_bokeh_color_bar_png(
    color_mapper: LinearColorMapper,
    title: str,
    output_path: str | Path,
) -> None:
    legend_plot = figure(
        width=140,
        height=320,
        toolbar_location=None,
        outline_line_color=None,
        background_fill_color="white",
    )
    legend_plot.axis.visible = False
    legend_plot.grid.visible = False
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=8,
        title=title,
    )
    legend_plot.add_layout(color_bar, "right")
    _export_bokeh_png(legend_plot, output_path)


def _export_bokeh_categorical_legend_png(
    items: list[tuple[str, str]],
    output_path: str | Path,
    *,
    orientation: str = "horizontal",
) -> None:
    from matplotlib.lines import Line2D

    if not items:
        return

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="#1f2937",
            markeredgewidth=0.8,
            markersize=8,
            label=label,
        )
        for label, color in items
    ]
    labels = [label for label, _ in items]
    ncol = len(items) if orientation == "horizontal" else 1
    figsize = (max(4.0, 1.6 * len(items)), 0.6) if orientation == "horizontal" else (3.0, 1.5)
    legend_fig, legend_axis = plt.subplots(figsize=figsize)
    legend_axis.axis("off")
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=FONTSIZE_DEFAULT,
        ncol=ncol,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    legend_fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(legend_fig)


def _add_map_color_bar(
    plot,
    color_mapper: LinearColorMapper,
    title: str,
    *,
    label_standoff: int = 8,
    separated_legend: bool = MAP_SEPARATED_LEGEND_DEFAULT,
) -> None:
    if separated_legend:
        return
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=label_standoff,
        title=title,
    )
    plot.add_layout(color_bar, "right")


def _resolve_qgis_output_dir(
    output_qgis_dir: str | Path | None,
    output_html: str | Path | None,
    folder_name: str,
) -> Path | None:
    if output_qgis_dir is not None:
        return Path(output_qgis_dir)
    if output_html is not None:
        return Path(output_html).parent / "for_QGIS" / folder_name
    return None


def _build_kge_qgis_node_attributes(
    station_ids: list[str],
    r_values: list[float],
    alpha_values: list[float],
    beta_values: list[float],
    kge_values: list[float],
) -> dict[str, dict[str, object]]:
    node_extra_attributes: dict[str, dict[str, object]] = {}
    for index, station_id in enumerate(station_ids):
        r_value = r_values[index]
        alpha_value = alpha_values[index]
        beta_value = beta_values[index]
        shares = _compute_kge_component_shares(r_value, alpha_value, beta_value)
        record: dict[str, object] = {
            "r": r_value,
            "alpha": alpha_value,
            "beta": beta_value,
            "KGE": kge_values[index],
        }
        if shares is None:
            record["r_contrib"] = float("nan")
            record["alpha_contrib"] = float("nan")
            record["beta_contrib"] = float("nan")
        else:
            record["r_contrib"] = shares[0]
            record["alpha_contrib"] = shares[1]
            record["beta_contrib"] = shares[2]
        node_extra_attributes[station_id] = record
    return node_extra_attributes


def _export_map_graph_for_qgis(
    *,
    output_qgis_dir: str | Path | None,
    output_html: str | Path | None,
    folder_name: str,
    gpkg_name: str,
    static_info_path: str | Path,
    station_ids: list[str],
    weighted_adj_matrix: pd.DataFrame,
    node_extra_attributes: dict[str, dict[str, object]],
) -> Path | None:
    qgis_dir = _resolve_qgis_output_dir(output_qgis_dir, output_html, folder_name)
    if qgis_dir is None:
        return None
    return export_graph_for_qgis(
        output_dir=qgis_dir,
        gpkg_name=gpkg_name,
        static_info_path=static_info_path,
        station_ids=station_ids,
        weighted_adj_matrix=weighted_adj_matrix,
        node_extra_attributes=node_extra_attributes,
    )


def _save_bokeh_html(plot, output_html: str | Path) -> None:
    from bokeh.io import save
    from bokeh.resources import INLINE

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    save(plot, filename=str(output_html), resources=INLINE, title=output_html.stem)


def _export_bokeh_png(plot, output_png: str | Path) -> None:
    try:
        from bokeh.io import export_png

        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        export_png(plot, filename=str(output_png))
    except Exception as exc:
        print(f"Warning: could not export map PNG to {output_png}: {exc}")


def _build_directed_edges(
    weighted_adj_matrix: pd.DataFrame,
    coord_lookup: dict[str, tuple[float, float]],
    *,
    arrow_offset: float = 2000.0,
) -> tuple[ColumnDataSource, list[float], list[float], list[float], list[float]]:
    edge_starts_x: list[float] = []
    edge_starts_y: list[float] = []
    edge_ends_x: list[float] = []
    edge_ends_y: list[float] = []
    edge_weights: list[float] = []
    edge_weights_km: list[float] = []
    edge_weights_label: list[str] = []
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_sources: list[str] = []
    edge_targets: list[str] = []

    station_ids = list(weighted_adj_matrix.index)
    for source_id in station_ids:
        for target_id in station_ids:
            weight = weighted_adj_matrix.at[source_id, target_id]
            if weight == 0:
                continue
            start_x, start_y = coord_lookup[source_id]
            end_x, end_y = coord_lookup[target_id]
            dx = end_x - start_x
            dy = end_y - start_y
            dist = math.hypot(dx, dy)
            if dist > 0:
                start_x = start_x + dx / dist * arrow_offset
                start_y = start_y + dy / dist * arrow_offset
                end_x = end_x - dx / dist * arrow_offset
                end_y = end_y - dy / dist * arrow_offset
            edge_starts_x.append(start_x)
            edge_starts_y.append(start_y)
            edge_ends_x.append(end_x)
            edge_ends_y.append(end_y)
            weight_value = float(weight)
            edge_weights.append(weight_value)
            weight_km = weight_value / 1000.0
            edge_weights_km.append(weight_km)
            edge_weights_label.append(f"{weight_km:.2f} km")
            edge_mid_x.append((start_x + end_x) / 2)
            edge_mid_y.append((start_y + end_y) / 2)
            edge_sources.append(source_id)
            edge_targets.append(target_id)

    edges_source = ColumnDataSource(
        {
            "x0": edge_starts_x,
            "y0": edge_starts_y,
            "x1": edge_ends_x,
            "y1": edge_ends_y,
            "weight": edge_weights,
            "weight_km": edge_weights_km,
            "weight_label": edge_weights_label,
            "mid_x": edge_mid_x,
            "mid_y": edge_mid_y,
            "source_id": edge_sources,
            "target_id": edge_targets,
        }
    )
    return edges_source, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y


def _add_graph_arrows(
    plot,
    edge_starts_x: list[float],
    edge_starts_y: list[float],
    edge_ends_x: list[float],
    edge_ends_y: list[float],
) -> None:
    for x0, y0, x1, y1 in zip(edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y):
        plot.add_layout(
            Arrow(
                end=NormalHead(size=12, fill_color="#93c5fd", line_color="#93c5fd"),
                line_color="#2a8cfb",
                line_alpha=0.4,
                line_width=4,
                x_start=x0,
                y_start=y0,
                x_end=x1,
                y_end=y1,
            )
        )


def plot_graph_error_map(
    weighted_adj_matrix: pd.DataFrame,
    static_info_path: str | Path,
    error_by_station: dict[str, float],
    error_metric: str = "NSE",
    output_html: str | Path | None = None,
    output_png: str | Path | None = None,
    output_qgis_dir: str | Path | None = None,
    show_errors: bool = True,
    show_edge_km: bool = True,
    show_plot: bool = False,
    separated_legend: bool = MAP_SEPARATED_LEGEND_DEFAULT,
) -> None:
    metric = _validate_error_metric(error_metric)
    if not isinstance(weighted_adj_matrix, pd.DataFrame):
        raise TypeError("weighted_adj_matrix must be a pandas DataFrame")
    if list(weighted_adj_matrix.index) != list(weighted_adj_matrix.columns):
        raise ValueError("weighted_adj_matrix index and columns must match")

    static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
    static_info_df = static_info_df.set_index("station_id")

    station_ids = list(weighted_adj_matrix.index)
    latitudes: list[float] = []
    longitudes: list[float] = []
    x_coords: list[float] = []
    y_coords: list[float] = []
    station_names: list[str] = []
    catchment_areas: list[float | None] = []
    elevations: list[float | None] = []
    agri_areas: list[float | None] = []
    forest_areas: list[float | None] = []
    shrub_areas: list[float | None] = []
    error_values: list[float] = []

    for station_id in station_ids:
        if station_id not in static_info_df.index:
            raise ValueError(f"Station {station_id} not found in static info")
        row = static_info_df.loc[station_id]
        lat = row.get("Latitude")
        lon = row.get("Longitude")
        if pd.isna(lat) or pd.isna(lon):
            raise ValueError(f"Missing latitude/longitude for station {station_id}")
        lat_value = float(lat)
        lon_value = float(lon)
        latitudes.append(lat_value)
        longitudes.append(lon_value)
        x_value, y_value = _lonlat_to_mercator(lon_value, lat_value)
        x_coords.append(x_value)
        y_coords.append(y_value)
        station_names.append(str(row.get("Station name") or ""))
        catchment_areas.append(
            float(row.get("Catchment area"))
            if not pd.isna(row.get("Catchment area"))
            else None
        )
        elevations.append(
            float(row.get("Elevation"))
            if not pd.isna(row.get("Elevation"))
            else None
        )
        agri_areas.append(
            float(row.get("Agricultural area (%)"))
            if not pd.isna(row.get("Agricultural area (%)"))
            else None
        )
        forest_areas.append(
            float(row.get("Forestal area (%)"))
            if not pd.isna(row.get("Forestal area (%)"))
            else None
        )
        shrub_areas.append(
            float(row.get("Shrub area (%)"))
            if not pd.isna(row.get("Shrub area (%)"))
            else None
        )
        error_value = error_by_station.get(station_id)
        error_values.append(float(error_value) if error_value is not None else float("nan"))

    error_labels = [
        _format_error_label(metric, value) for value in error_values
    ]

    error_array = np.array([value for value in error_values if np.isfinite(value)], dtype=np.float64)
    if error_array.size == 0:
        error_low, error_high = 0.0, 1.0
    else:
        error_low = float(error_array.min())
        error_high = float(error_array.max())
        if math.isclose(error_low, error_high):
            error_high = error_low + 1.0

    color_mapper = LinearColorMapper(
        palette=Viridis256,
        low=error_low,
        high=error_high,
        nan_color="#9ca3af",
    )

    nodes_source = ColumnDataSource(
        {
            "station_id": station_ids,
            "lat": latitudes,
            "lon": longitudes,
            "x": x_coords,
            "y": y_coords,
            "station_name": station_names,
            "catchment_area": catchment_areas,
            "elevation": elevations,
            "agri_area": agri_areas,
            "forest_area": forest_areas,
            "shrub_area": shrub_areas,
            "error_value": error_values,
            "error_label": error_labels,
        }
    )

    coord_lookup = dict(zip(station_ids, zip(x_coords, y_coords)))
    edges_source, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y = _build_directed_edges(
        weighted_adj_matrix,
        coord_lookup,
    )

    plot = figure(
        title=f"Station graph with node {metric}",
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_axis_label="Longitude",
        y_axis_label="Latitude",
        width=900,
        height=700,
    )
    tile_source = WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png",
        attribution="&copy; OpenStreetMap contributors &copy; CARTO",
    )
    plot.add_tile(tile_source)

    plot.segment(
        "x0",
        "y0",
        "x1",
        "y1",
        source=edges_source,
        line_width=4,
        line_alpha=0.4,
        line_color="#93c5fd",
        level="underlay",
    )
    node_renderer = plot.circle(
        "x",
        "y",
        size=16,
        source=nodes_source,
        line_color="#1f2937",
        fill_color={"field": "error_value", "transform": color_mapper},
        hover_color="#b91c1c",
    )

    _add_graph_arrows(plot, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y)

    node_labels = LabelSet(
        x="x",
        y="y",
        text="station_id",
        source=nodes_source,
        x_offset=6,
        y_offset=6,
        text_color="#000000",
        level="overlay",
    )
    plot.add_layout(node_labels)

    if show_errors:
        error_value_labels = LabelSet(
            x="x",
            y="y",
            text="error_label",
            source=nodes_source,
            x_offset=0,
            y_offset=-10,
            text_font_size="9pt",
            text_color="#111827",
            text_align="center",
            text_baseline="top",
            level="overlay",
        )
        plot.add_layout(error_value_labels)

    if show_edge_km:
        edge_labels = LabelSet(
            x="mid_x",
            y="mid_y",
            text="weight_label",
            source=edges_source,
            text_font_size="8pt",
            text_color="#000000",
            level="overlay",
        )
        plot.add_layout(edge_labels)

    error_value_format = "0.0000" if metric in ("NSE", "KGE") else "0.000"
    node_hover = HoverTool(
        renderers=[node_renderer],
        tooltips=[
            ("Station", "@station_id"),
            ("Name", "@station_name"),
            (metric, f"@error_value{{{error_value_format}}}"),
            ("Catchment area", "@catchment_area{0.00}"),
            ("Elevation", "@elevation{0.00}"),
            ("Agricultural", "@agri_area{0.00}"),
            ("Forestal", "@forest_area{0.00}"),
            ("Shrub", "@shrub_area{0.00}"),
        ],
    )
    plot.add_tools(node_hover)

    _apply_map_view_padding(plot, x_coords, y_coords)
    _add_map_color_bar(
        plot,
        color_mapper,
        metric,
        separated_legend=separated_legend,
    )

    if output_html is not None:
        _save_bokeh_html(plot, output_html)
    if output_png is not None:
        _export_bokeh_png(plot, output_png)
        if separated_legend:
            _export_bokeh_color_bar_png(
                color_mapper,
                metric,
                _map_color_bar_legend_png_path(output_png, error_low, error_high),
            )
    _export_map_graph_for_qgis(
        output_qgis_dir=output_qgis_dir,
        output_html=output_html,
        folder_name=f"error_map_{metric}",
        gpkg_name=f"error_map_{metric}.gpkg",
        static_info_path=static_info_path,
        station_ids=station_ids,
        weighted_adj_matrix=weighted_adj_matrix,
        node_extra_attributes={
            station_id: {metric: error_values[index]}
            for index, station_id in enumerate(station_ids)
        },
    )
    if show_plot:
        show(plot)


def plot_KGE_separated_map(
    weighted_adj_matrix: pd.DataFrame,
    static_info_path: str | Path,
    *,
    station_frames: dict[str, pd.DataFrame] | None = None,
    kge_components_by_station: dict[str, tuple[float, float, float]] | None = None,
    output_html: str | Path | None = None,
    output_png: str | Path | None = None,
    output_qgis_dir: str | Path | None = None,
    show_edge_km: bool = True,
    show_plot: bool = False,
    separated_legend: bool = MAP_SEPARATED_LEGEND_DEFAULT,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> None:
    """
    Plot a station graph map where each node is a pie chart of KGE component
    contributions from r, alpha, and beta.
    """
    if station_frames is None and kge_components_by_station is None:
        raise ValueError("Provide station_frames or kge_components_by_station")
    if not isinstance(weighted_adj_matrix, pd.DataFrame):
        raise TypeError("weighted_adj_matrix must be a pandas DataFrame")
    if list(weighted_adj_matrix.index) != list(weighted_adj_matrix.columns):
        raise ValueError("weighted_adj_matrix index and columns must match")

    if kge_components_by_station is None:
        kge_components_by_station = compute_kge_components_by_station_from_frames(
            station_frames,
            observed_col=observed_col,
            predicted_col=predicted_col,
        )

    static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
    static_info_df = static_info_df.set_index("station_id")

    station_ids = list(weighted_adj_matrix.index)
    latitudes: list[float] = []
    longitudes: list[float] = []
    x_coords: list[float] = []
    y_coords: list[float] = []
    station_names: list[str] = []
    catchment_areas: list[float | None] = []
    elevations: list[float | None] = []
    agri_areas: list[float | None] = []
    forest_areas: list[float | None] = []
    shrub_areas: list[float | None] = []
    r_values: list[float] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    kge_values: list[float] = []
    component_labels: list[str] = []

    for station_id in station_ids:
        if station_id not in static_info_df.index:
            raise ValueError(f"Station {station_id} not found in static info")
        row = static_info_df.loc[station_id]
        lat = row.get("Latitude")
        lon = row.get("Longitude")
        if pd.isna(lat) or pd.isna(lon):
            raise ValueError(f"Missing latitude/longitude for station {station_id}")
        lat_value = float(lat)
        lon_value = float(lon)
        latitudes.append(lat_value)
        longitudes.append(lon_value)
        x_value, y_value = _lonlat_to_mercator(lon_value, lat_value)
        x_coords.append(x_value)
        y_coords.append(y_value)
        station_names.append(str(row.get("Station name") or ""))
        catchment_areas.append(
            float(row.get("Catchment area"))
            if not pd.isna(row.get("Catchment area"))
            else None
        )
        elevations.append(
            float(row.get("Elevation"))
            if not pd.isna(row.get("Elevation"))
            else None
        )
        agri_areas.append(
            float(row.get("Agricultural area (%)"))
            if not pd.isna(row.get("Agricultural area (%)"))
            else None
        )
        forest_areas.append(
            float(row.get("Forestal area (%)"))
            if not pd.isna(row.get("Forestal area (%)"))
            else None
        )
        shrub_areas.append(
            float(row.get("Shrub area (%)"))
            if not pd.isna(row.get("Shrub area (%)"))
            else None
        )
        r_value, alpha_value, beta_value = kge_components_by_station.get(
            station_id,
            (float("nan"), float("nan"), float("nan")),
        )
        r_values.append(float(r_value))
        alpha_values.append(float(alpha_value))
        beta_values.append(float(beta_value))
        if all(np.isfinite(value) for value in (r_value, alpha_value, beta_value)):
            kge_value = float(
                1.0
                - np.sqrt(
                    (r_value - 1.0) ** 2
                    + (alpha_value - 1.0) ** 2
                    + (beta_value - 1.0) ** 2
                )
            )
        else:
            kge_value = float("nan")
        kge_values.append(kge_value)
        component_labels.append(_format_kge_components_label(r_value, alpha_value, beta_value))

    nodes_source = ColumnDataSource(
        {
            "station_id": station_ids,
            "lat": latitudes,
            "lon": longitudes,
            "x": x_coords,
            "y": y_coords,
            "station_name": station_names,
            "catchment_area": catchment_areas,
            "elevation": elevations,
            "agri_area": agri_areas,
            "forest_area": forest_areas,
            "shrub_area": shrub_areas,
            "r_value": r_values,
            "alpha_value": alpha_values,
            "beta_value": beta_values,
            "kge_value": kge_values,
            "component_label": component_labels,
        }
    )

    coord_lookup = dict(zip(station_ids, zip(x_coords, y_coords)))
    wedge_source, gray_source = _build_kge_pie_wedge_source(
        station_ids,
        coord_lookup,
        kge_components_by_station,
    )
    edges_source, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y = _build_directed_edges(
        weighted_adj_matrix,
        coord_lookup,
    )

    plot = figure(
        title="Station graph with KGE component contributions",
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_axis_label="Longitude",
        y_axis_label="Latitude",
        width=900,
        height=700,
        min_border_top=50,
        min_border_bottom=90,
        min_border_left=50,
        min_border_right=50,
    )
    tile_source = WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png",
        attribution="&copy; OpenStreetMap contributors &copy; CARTO",
    )
    plot.add_tile(tile_source)

    plot.segment(
        "x0",
        "y0",
        "x1",
        "y1",
        source=edges_source,
        line_width=4,
        line_alpha=0.4,
        line_color="#93c5fd",
        level="underlay",
    )

    if len(gray_source.data["x"]) > 0:
        plot.circle(
            "x",
            "y",
            size=KGE_PIE_RADIUS * 2,
            source=gray_source,
            line_color="#1f2937",
            fill_color=KGE_PIE_FALLBACK_COLOR,
            level="overlay",
        )
    if len(wedge_source.data["x"]) > 0:
        plot.wedge(
            "x",
            "y",
            KGE_PIE_RADIUS,
            "start_angle",
            "end_angle",
            source=wedge_source,
            radius_units="screen",
            line_color="#1f2937",
            fill_color="color",
            level="overlay",
        )

    hover_renderer = plot.circle(
        "x",
        "y",
        size=KGE_PIE_RADIUS * 2 + 4,
        source=nodes_source,
        fill_alpha=0,
        line_alpha=0,
        level="overlay",
    )

    _add_graph_arrows(plot, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y)

    node_labels = LabelSet(
        x="x",
        y="y",
        text="station_id",
        source=nodes_source,
        x_offset=0,
        y_offset=KGE_MAP_STATION_LABEL_Y_OFFSET,
        text_align="center",
        text_baseline="bottom",
        text_color="#000000",
        level="overlay",
    )
    plot.add_layout(node_labels)

    component_value_labels = LabelSet(
        x="x",
        y="y",
        text="component_label",
        source=nodes_source,
        x_offset=0,
        y_offset=KGE_MAP_COMPONENT_LABEL_Y_OFFSET,
        text_font_size="9pt",
        text_color="#111827",
        text_align="center",
        text_baseline="top",
        level="overlay",
    )
    plot.add_layout(component_value_labels)

    if show_edge_km:
        edge_labels = LabelSet(
            x="mid_x",
            y="mid_y",
            text="weight_label",
            source=edges_source,
            text_font_size="8pt",
            text_color="#000000",
            level="overlay",
        )
        plot.add_layout(edge_labels)

    node_hover = HoverTool(
        renderers=[hover_renderer],
        tooltips=[
            ("Station", "@station_id"),
            ("Name", "@station_name"),
            ("r", "@r_value{0.0000}"),
            ("α", "@alpha_value{0.0000}"),
            ("β", "@beta_value{0.0000}"),
            ("KGE", "@kge_value{0.0000}"),
            ("Catchment area", "@catchment_area{0.00}"),
            ("Elevation", "@elevation{0.00}"),
            ("Agricultural", "@agri_area{0.00}"),
            ("Forestal", "@forest_area{0.00}"),
            ("Shrub", "@shrub_area{0.00}"),
        ],
    )
    plot.add_tools(node_hover)

    _apply_map_view_padding(
        plot,
        x_coords,
        y_coords,
        padding_fraction=KGE_MAP_VIEW_PADDING_FRACTION,
    )

    kge_legend_items = [
        ("r contribution", KGE_COMPONENT_COLORS["r"]),
        ("α contribution", KGE_COMPONENT_COLORS["alpha"]),
        ("β contribution", KGE_COMPONENT_COLORS["beta"]),
    ]
    if not separated_legend:
        legend_anchor_x = plot.x_range.start - (plot.x_range.end - plot.x_range.start) * 0.5
        legend_anchor_y = plot.y_range.start - (plot.y_range.end - plot.y_range.start) * 0.5
        r_legend = plot.scatter(
            [legend_anchor_x],
            [legend_anchor_y],
            size=12,
            fill_color=KGE_COMPONENT_COLORS["r"],
            line_color="#1f2937",
            level="overlay",
        )
        alpha_legend = plot.scatter(
            [legend_anchor_x],
            [legend_anchor_y],
            size=12,
            fill_color=KGE_COMPONENT_COLORS["alpha"],
            line_color="#1f2937",
            level="overlay",
        )
        beta_legend = plot.scatter(
            [legend_anchor_x],
            [legend_anchor_y],
            size=12,
            fill_color=KGE_COMPONENT_COLORS["beta"],
            line_color="#1f2937",
            level="overlay",
        )
        legend = Legend(
            items=[
                LegendItem(label="r contribution", renderers=[r_legend]),
                LegendItem(label="α contribution", renderers=[alpha_legend]),
                LegendItem(label="β contribution", renderers=[beta_legend]),
            ],
            location="bottom_center",
            orientation="horizontal",
            click_policy="hide",
        )
        plot.add_layout(legend, "below")

    if output_html is not None:
        _save_bokeh_html(plot, output_html)
    if output_png is not None:
        _export_bokeh_png(plot, output_png)
        if separated_legend:
            _export_bokeh_categorical_legend_png(
                kge_legend_items,
                _map_categorical_legend_png_path(output_png),
                orientation="horizontal",
            )
    _export_map_graph_for_qgis(
        output_qgis_dir=output_qgis_dir,
        output_html=output_html,
        folder_name="KGE_separated",
        gpkg_name="KGE_separated.gpkg",
        static_info_path=static_info_path,
        station_ids=station_ids,
        weighted_adj_matrix=weighted_adj_matrix,
        node_extra_attributes=_build_kge_qgis_node_attributes(
            station_ids,
            r_values,
            alpha_values,
            beta_values,
            kge_values,
        ),
    )
    if show_plot:
        show(plot)


def comparison_NSE_conchi( ###  TODO TODO TODO REVISAR MÉTODO ENTERO
    weighted_adj_matrix: pd.DataFrame,
    static_info_path: str | Path,
    *,
    station_frames: dict[str, pd.DataFrame] | None = None,
    nse_by_station: dict[str, float] | None = None,
    conchi_nse_path: str | Path = DEFAULT_CONCHI_NSE_PATH,
    conchi_model: ConchiModel = "LSTM",
    conchi_scenario: ConchiScenario = "TS2",
    output_html: str | Path | None = None,
    output_png: str | Path | None = None,
    output_qgis_dir: str | Path | None = None,
    show_edge_km: bool = True,
    show_plot: bool = False,
    separated_legend: bool = MAP_SEPARATED_LEGEND_DEFAULT,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> dict[str, float | None]:
    """
    Compare model NSE against Conchi's published NSE values on a graph map.

    Node color encodes model NSE minus Conchi NSE (blue = model better, red = Conchi
    better, white = equal). Stations missing from Conchi's data are shown in gray.
    """
    if station_frames is None and nse_by_station is None:
        raise ValueError("Provide station_frames or nse_by_station")
    if not isinstance(weighted_adj_matrix, pd.DataFrame):
        raise TypeError("weighted_adj_matrix must be a pandas DataFrame")
    if list(weighted_adj_matrix.index) != list(weighted_adj_matrix.columns):
        raise ValueError("weighted_adj_matrix index and columns must match")

    if nse_by_station is None:
        nse_by_station = compute_error_by_station_from_frames(
            station_frames,
            "NSE",
            observed_col=observed_col,
            predicted_col=predicted_col,
        )

    conchi_nse_lookup = _load_conchi_nse_by_station(conchi_nse_path, conchi_model, conchi_scenario)

    static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
    static_info_df = static_info_df.set_index("station_id")

    station_ids = list(weighted_adj_matrix.index)
    latitudes: list[float] = []
    longitudes: list[float] = []
    x_coords: list[float] = []
    y_coords: list[float] = []
    station_names: list[str] = []
    catchment_areas: list[float | None] = []
    elevations: list[float | None] = []
    agri_areas: list[float | None] = []
    forest_areas: list[float | None] = []
    shrub_areas: list[float | None] = []
    model_nse_values: list[float] = []
    conchi_nse_values: list[float] = []
    nse_difference_values: list[float] = []
    difference_labels: list[str] = []

    nse_differences: dict[str, float | None] = {}

    print(
        f"NSE comparison vs Conchi ({conchi_scenario}, {conchi_model}) "
        f"[model NSE - Conchi NSE]:"
    )
    for station_id in station_ids:
        if station_id not in static_info_df.index:
            raise ValueError(f"Station {station_id} not found in static info")
        row = static_info_df.loc[station_id]
        lat = row.get("Latitude")
        lon = row.get("Longitude")
        if pd.isna(lat) or pd.isna(lon):
            raise ValueError(f"Missing latitude/longitude for station {station_id}")
        lat_value = float(lat)
        lon_value = float(lon)
        latitudes.append(lat_value)
        longitudes.append(lon_value)
        x_value, y_value = _lonlat_to_mercator(lon_value, lat_value)
        x_coords.append(x_value)
        y_coords.append(y_value)
        station_names.append(str(row.get("Station name") or ""))
        catchment_areas.append(
            float(row.get("Catchment area"))
            if not pd.isna(row.get("Catchment area"))
            else None
        )
        elevations.append(
            float(row.get("Elevation"))
            if not pd.isna(row.get("Elevation"))
            else None
        )
        agri_areas.append(
            float(row.get("Agricultural area (%)"))
            if not pd.isna(row.get("Agricultural area (%)"))
            else None
        )
        forest_areas.append(
            float(row.get("Forestal area (%)"))
            if not pd.isna(row.get("Forestal area (%)"))
            else None
        )
        shrub_areas.append(
            float(row.get("Shrub area (%)"))
            if not pd.isna(row.get("Shrub area (%)"))
            else None
        )

        model_nse = nse_by_station.get(station_id)
        model_nse_float = float(model_nse) if model_nse is not None else float("nan")
        conchi_nse = conchi_nse_lookup.get(_normalize_station_id_for_conchi(station_id))
        if conchi_nse is None:
            difference = None
            difference_float = float("nan")
            print(f"  {station_id}: model={model_nse_float:.4f}, Conchi=N/A, diff=N/A")
        else:
            difference = model_nse_float - conchi_nse
            difference_float = float(difference)
            print(
                f"  {station_id}: model={model_nse_float:.4f}, "
                f"Conchi={conchi_nse:.4f}, diff={difference:+.4f}"
            )

        nse_differences[station_id] = difference
        model_nse_values.append(model_nse_float)
        conchi_nse_values.append(float(conchi_nse) if conchi_nse is not None else float("nan"))
        nse_difference_values.append(difference_float)
        difference_labels.append(_format_signed_difference(difference))

    diff_low, diff_high = _symmetric_diverging_bounds(nse_difference_values)
    color_mapper = LinearColorMapper(
        palette=_red_white_blue_palette(),
        low=diff_low,
        high=diff_high,
        nan_color=MISSING_CONCHI_COLOR,
    )

    nodes_source = ColumnDataSource(
        {
            "station_id": station_ids,
            "lat": latitudes,
            "lon": longitudes,
            "x": x_coords,
            "y": y_coords,
            "station_name": station_names,
            "catchment_area": catchment_areas,
            "elevation": elevations,
            "agri_area": agri_areas,
            "forest_area": forest_areas,
            "shrub_area": shrub_areas,
            "model_nse": model_nse_values,
            "conchi_nse": conchi_nse_values,
            "nse_difference": nse_difference_values,
            "difference_label": difference_labels,
        }
    )

    coord_lookup = dict(zip(station_ids, zip(x_coords, y_coords)))
    edges_source, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y = _build_directed_edges(
        weighted_adj_matrix,
        coord_lookup,
    )

    plot = figure(
        title=(
            f"NSE difference vs Conchi ({conchi_scenario}, {conchi_model}) "
            f"[model - Conchi; blue = model better]"
        ),
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_axis_label="Longitude",
        y_axis_label="Latitude",
        width=900,
        height=700,
    )
    tile_source = WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png",
        attribution="&copy; OpenStreetMap contributors &copy; CARTO",
    )
    plot.add_tile(tile_source)

    plot.segment(
        "x0",
        "y0",
        "x1",
        "y1",
        source=edges_source,
        line_width=4,
        line_alpha=0.4,
        line_color="#93c5fd",
        level="underlay",
    )
    node_renderer = plot.circle(
        "x",
        "y",
        size=16,
        source=nodes_source,
        line_color="#1f2937",
        fill_color={"field": "nse_difference", "transform": color_mapper},
        hover_color="#b91c1c",
    )

    _add_graph_arrows(plot, edge_starts_x, edge_starts_y, edge_ends_x, edge_ends_y)

    node_labels = LabelSet(
        x="x",
        y="y",
        text="station_id",
        source=nodes_source,
        x_offset=6,
        y_offset=6,
        text_color="#000000",
        level="overlay",
    )
    plot.add_layout(node_labels)

    difference_value_labels = LabelSet(
        x="x",
        y="y",
        text="difference_label",
        source=nodes_source,
        x_offset=-18,
        y_offset=-18,
        text_font_size="9pt",
        text_color="#111827",
        text_align="center",
        level="overlay",
    )
    plot.add_layout(difference_value_labels)

    if show_edge_km:
        edge_labels = LabelSet(
            x="mid_x",
            y="mid_y",
            text="weight_label",
            source=edges_source,
            text_font_size="8pt",
            text_color="#000000",
            level="overlay",
        )
        plot.add_layout(edge_labels)

    node_hover = HoverTool(
        renderers=[node_renderer],
        tooltips=[
            ("Station", "@station_id"),
            ("Name", "@station_name"),
            ("Model NSE", "@model_nse{0.0000}"),
            ("Conchi NSE", "@conchi_nse{0.0000}"),
            ("Difference", "@difference_label"),
            ("Catchment area", "@catchment_area{0.00}"),
            ("Elevation", "@elevation{0.00}"),
            ("Agricultural", "@agri_area{0.00}"),
            ("Forestal", "@forest_area{0.00}"),
            ("Shrub", "@shrub_area{0.00}"),
        ],
    )
    plot.add_tools(node_hover)

    missing_renderer = plot.scatter(
        [],
        [],
        size=12,
        fill_color=MISSING_CONCHI_COLOR,
        line_color="#1f2937",
    )
    _apply_map_view_padding(plot, x_coords, y_coords)
    color_bar_title = "NSE diff\n(GNN_LSTM - LSTM)"
    _add_map_color_bar(
        plot,
        color_mapper,
        color_bar_title,
        separated_legend=separated_legend,
    )
    node_legend_items = [
        ("Colored nodes: model NSE - Conchi NSE", "#3182bd"),
        ("Gray: station not in Conchi data", MISSING_CONCHI_COLOR),
    ]
    if not separated_legend:
        legend = Legend(
            items=[
                LegendItem(label=node_legend_items[0][0], renderers=[node_renderer]),
                LegendItem(label=node_legend_items[1][0], renderers=[missing_renderer]),
            ],
            location="bottom_center",
            orientation="horizontal",
            click_policy="hide",
        )
        plot.add_layout(legend, "below")

    if output_html is not None:
        _save_bokeh_html(plot, output_html)
    if output_png is not None:
        _export_bokeh_png(plot, output_png)
        if separated_legend:
            _export_bokeh_color_bar_png(
                color_mapper,
                color_bar_title,
                _map_color_bar_legend_png_path(output_png, diff_low, diff_high),
            )
            _export_bokeh_categorical_legend_png(
                node_legend_items,
                _map_categorical_legend_png_path(output_png),
                orientation="horizontal",
            )
    qgis_folder_name = f"NSE_conchi_{conchi_model}_{conchi_scenario}"
    _export_map_graph_for_qgis(
        output_qgis_dir=output_qgis_dir,
        output_html=output_html,
        folder_name=qgis_folder_name,
        gpkg_name=f"NSE_conchi_{conchi_model}_{conchi_scenario}.gpkg",
        static_info_path=static_info_path,
        station_ids=station_ids,
        weighted_adj_matrix=weighted_adj_matrix,
        node_extra_attributes={
            station_id: {
                "model_nse": model_nse_values[index],
                "conchi_nse": conchi_nse_values[index],
                "nse_difference": nse_difference_values[index],
                "conchi_model": conchi_model,
                "conchi_scenario": conchi_scenario,
            }
            for index, station_id in enumerate(station_ids)
        },
    )
    if show_plot:
        show(plot)

    return nse_differences












### TODO INIT REVISAR CHAT
def _format_return_period_label(return_period: float) -> str:
    label = RETURN_PERIOD_LABELS.get(return_period)
    if label is None:
        label = RETURN_PERIOD_LABELS.get(float(return_period))
    if label is None:
        label = f"T{return_period:g}"
    return label


def _format_return_period_axis_label(return_period: float) -> str:
    return rf"$T_{{{return_period:g}}}$"


def _return_period_model_order(
    gnn_model_label: str,
    *,
    include_mc_lstm: bool = True,
) -> list[str]:
    model_order = [gnn_model_label, "LSTM"]
    if include_mc_lstm:
        model_order.append("MC-LSTM")
    return model_order


def _apply_axis_font_size(axis: plt.Axes, fontsize: float) -> None:
    axis.xaxis.label.set_size(fontsize)
    axis.yaxis.label.set_size(fontsize)
    axis.tick_params(labelsize=fontsize)
    title = axis.get_title()
    if title:
        axis.set_title(title, fontsize=fontsize)


def _save_separated_figure_legend(
    handles: list,
    legend_path: Path,
    *,
    fontsize: float = FONTSIZE_DEFAULT,
    ncol: int | None = None,
) -> None:
    labels = [handle.get_label() for handle in handles]
    if not labels:
        return
    legend_fig, legend_axis = plt.subplots(figsize=(4, 0.5))
    legend_axis.axis("off")
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=fontsize,
        ncol=ncol if ncol is not None else len(labels),
    )
    legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight")
    plt.close(legend_fig)


def _load_return_period_thresholds_lookup(
    return_periods_path: str | Path,
) -> dict[str, dict[float, float]]:
    """
    Load per-station streamflow lower bounds for each return period.

    The CSV is expected to contain `station_id` plus one column per return period
    (for example `0.5`, `1`, `2`, `5`, `10`).
    """
    path = Path(return_periods_path)
    thresholds_df = pd.read_csv(path, dtype={"station_id": str})
    if "station_id" not in thresholds_df.columns:
        raise ValueError(f"Return-period thresholds file must include station_id: {path}")

    lookup: dict[str, dict[float, float]] = {}
    for _, row in thresholds_df.iterrows():
        station_id = _normalize_station_id_for_conchi(str(row["station_id"]))
        station_thresholds: dict[float, float] = {}
        for column_name in thresholds_df.columns:
            if str(column_name).strip().lower() == "station_id":
                continue
            try:
                period_key = float(str(column_name).strip())
            except ValueError:
                continue
            threshold_value = row[column_name]
            if pd.isna(threshold_value):
                continue
            station_thresholds[period_key] = float(threshold_value)
        if station_thresholds:
            lookup[station_id] = station_thresholds
    return lookup


def _get_station_return_period_thresholds(
    thresholds_lookup: dict[str, dict[float, float]],
    station_id: str,
) -> dict[str, float]:
    return thresholds_lookup.get(_normalize_station_id_for_conchi(station_id), {})


def _get_return_period_threshold(
    thresholds: dict[float, float],
    return_period: float,
) -> float | None:
    target_period = float(return_period)
    direct_value = thresholds.get(target_period)
    if direct_value is not None and np.isfinite(direct_value):
        return float(direct_value)
    for period_key, threshold_value in thresholds.items():
        if float(period_key) == target_period and np.isfinite(threshold_value):
            return float(threshold_value)
    return None


def _describe_return_period_event_bin(
    return_period: float,
    analysis_return_periods: list[float],
    thresholds: dict[float, float],
) -> str:
    analysis_sorted = sorted(float(period) for period in analysis_return_periods)
    normalized_return_period = float(return_period)
    period_label = _format_return_period_label(normalized_return_period)
    lower = _get_return_period_threshold(thresholds, normalized_return_period)
    if lower is None:
        return f"missing threshold column for {period_label} in summary return periods file"

    if normalized_return_period == analysis_sorted[-1]:
        return f"{period_label}: observed streamflow >= {lower:.4f}"

    if normalized_return_period == analysis_sorted[0]:
        if len(analysis_sorted) == 1:
            return f"{period_label}: observed streamflow >= {lower:.4f}"
        upper = _get_return_period_threshold(thresholds, analysis_sorted[1])
        if upper is None:
            return (
                f"{period_label}: observed streamflow >= {lower:.4f} "
                f"(missing upper threshold {_format_return_period_label(analysis_sorted[1])})"
            )
        return f"{period_label}: {lower:.4f} <= observed streamflow < {upper:.4f}"

    period_idx = analysis_sorted.index(normalized_return_period)
    upper = _get_return_period_threshold(thresholds, analysis_sorted[period_idx + 1])
    if upper is None:
        next_label = _format_return_period_label(analysis_sorted[period_idx + 1])
        return (
            f"{period_label}: observed streamflow >= {lower:.4f} "
            f"(missing upper threshold {next_label})"
        )
    return f"{period_label}: {lower:.4f} <= observed streamflow < {upper:.4f}"


def _mask_observed_for_return_period(
    observed: np.ndarray,
    return_period: float,
    analysis_return_periods: list[float],
    thresholds: dict[float, float],
) -> np.ndarray:
    """Build an event mask using observed streamflow only."""
    analysis_sorted = sorted(float(period) for period in analysis_return_periods)
    normalized_return_period = float(return_period)
    if normalized_return_period not in analysis_sorted:
        return np.zeros(observed.shape, dtype=bool)

    lower = _get_return_period_threshold(thresholds, normalized_return_period)
    if lower is None:
        return np.zeros(observed.shape, dtype=bool)

    if normalized_return_period == analysis_sorted[0]:
        if len(analysis_sorted) == 1:
            return observed >= lower
        upper = _get_return_period_threshold(thresholds, analysis_sorted[1])
        if upper is None:
            return observed >= lower
        return (observed >= lower) & (observed < upper)

    if normalized_return_period == analysis_sorted[-1]:
        return observed >= lower

    period_idx = analysis_sorted.index(normalized_return_period)
    upper = _get_return_period_threshold(thresholds, analysis_sorted[period_idx + 1])
    if upper is None:
        return observed >= lower
    return (observed >= lower) & (observed < upper)


def _compute_gnn_nrmse_for_return_period(
    frame: pd.DataFrame,
    *,
    return_period: float,
    analysis_return_periods: list[float],
    station_thresholds: dict[float, float],
    test_start_date: pd.Timestamp | None,
    examine_train_test: bool = True,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> tuple[float, str | None]:
    """
    Compute GNN nRMSE for one station and return period.

    Event-day selection uses observed streamflow only. Predictions are evaluated
    on those same days when computing nRMSE.

    When examine_train_test is True (default), peaks are evaluated over the full
    time series. When False, only days on or after test_start_date are used.
    """
    period_label = _format_return_period_label(return_period)
    day_scope = "train+test" if examine_train_test else "test"
    day_label = "day(s)" if examine_train_test else "test day(s)"
    if frame.empty:
        return float("nan"), "missing prediction frame for station"

    if not station_thresholds:
        return (
            float("nan"),
            "station not found in summary return periods file",
        )

    lower = _get_return_period_threshold(station_thresholds, return_period)
    if lower is None:
        return (
            float("nan"),
            f"missing threshold column for {period_label} in summary return periods file",
        )

    event_bin_description = _describe_return_period_event_bin(
        return_period,
        analysis_return_periods,
        station_thresholds,
    )

    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"])
    evaluation_frame = working
    if not examine_train_test and test_start_date is not None:
        evaluation_frame = working[working["date"] >= test_start_date]
    if evaluation_frame.empty:
        return float("nan"), f"no days available in {day_scope} period"

    observed = evaluation_frame[observed_col].to_numpy(dtype=np.float64)
    predicted = evaluation_frame[predicted_col].to_numpy(dtype=np.float64)
    event_mask = _mask_observed_for_return_period(
        observed,
        return_period,
        analysis_return_periods,
        station_thresholds,
    )
    event_day_count = int(event_mask.sum())
    if event_day_count == 0:
        return (
            float("nan"),
            f"no {day_scope} days matching event bin ({event_bin_description})",
        )

    event_observed = observed[event_mask]
    event_predicted = predicted[event_mask]
    finite_observed, finite_predicted = _paired_finite_arrays(event_observed, event_predicted)
    if finite_observed.size == 0:
        return (
            float("nan"),
            (
                f"{event_day_count} {day_label} matched event bin ({event_bin_description}) "
                "but none had finite observed and predicted values"
            ),
        )

    if float(finite_observed.mean()) == 0.0:
        return (
            float("nan"),
            (
                f"{finite_observed.size} {day_label} in event bin ({event_bin_description}) "
                "but mean observed streamflow is zero"
            ),
        )

    nrmse_value = compute_nrmse(finite_observed, finite_predicted)
    if not np.isfinite(nrmse_value):
        return (
            float("nan"),
            (
                f"{finite_observed.size} {day_label} in event bin ({event_bin_description}) "
                "but nRMSE could not be computed"
            ),
        )
    return float(nrmse_value), None


def _report_missing_gnn_return_period_points(
    station_ids: list[str],
    return_periods: list[float],
    gnn_values: dict[float, dict[str, float]],
    missing_reasons: dict[float, dict[str, str]],
    *,
    gnn_model_label: str,
) -> None:
    print(f"\nGNN return-period nRMSE warnings ({gnn_model_label}):")
    reported_any = False
    for station_id in station_ids:
        for return_period in return_periods:
            value = _lookup_gnn_nrmse_value(gnn_values, return_period, station_id)
            if value is not None:
                continue
            reported_any = True
            period_label = _format_return_period_label(return_period)
            reason = missing_reasons.get(float(return_period), {}).get(
                station_id,
                "unknown reason",
            )
            print(f"  Station {station_id} {period_label}: {reason}")
    if not reported_any:
        print("  None")


def _lookup_gnn_nrmse_value(
    gnn_values: dict[float, dict[str, float]],
    return_period: float,
    station_id: str,
) -> float | None:
    period_key = float(return_period)
    station_values = gnn_values.get(period_key)
    if station_values is None:
        for candidate_period, candidate_values in gnn_values.items():
            if float(candidate_period) == period_key:
                station_values = candidate_values
                break
    if station_values is None:
        return None
    value = station_values.get(station_id)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _conchi_source_file_matches_training_scenario(
    source_file: str | float | None,
    model_prefix: str,
    training_scenario: str,
) -> bool:
    if source_file is None or pd.isna(source_file):
        return False
    expected_prefix = f"{model_prefix}-{training_scenario}-"
    return str(source_file).startswith(expected_prefix)


def _conchi_model_prefix_to_display_name(model_prefix: str) -> str | None:
    normalized = model_prefix.strip().lower()
    if normalized == "lstm":
        return "LSTM"
    if normalized == "mclstm":
        return "MC-LSTM"
    return None


def _load_conchi_nrmse_lookup(
    conchi_nrmse_path: str | Path,
    *,
    training_scenario: ConchiTrainingScenario = "C2",
) -> dict[str, dict[str, dict[float, float]]]:
    path = Path(conchi_nrmse_path)
    conchi_df = pd.read_csv(path, dtype={"station_id": str})
    required_columns = {"station_id", "model", "source_file", *RETURN_PERIOD_VALUE_TO_COLUMN.values()}
    missing_columns = required_columns.difference(conchi_df.columns)
    if missing_columns:
        raise ValueError(f"Conchi return-period nRMSE file missing columns: {sorted(missing_columns)}")

    lookup: dict[str, dict[str, dict[float, float]]] = {}
    for _, row in conchi_df.iterrows():
        model_prefix = str(row["model"]).strip().lower()
        display_model = _conchi_model_prefix_to_display_name(model_prefix)
        if display_model is None:
            continue
        if not _conchi_source_file_matches_training_scenario(
            row.get("source_file"),
            model_prefix,
            training_scenario,
        ):
            continue

        station_id = _normalize_station_id_for_conchi(str(row["station_id"]))
        period_values: dict[float, float] = {}
        for return_period, column_name in RETURN_PERIOD_VALUE_TO_COLUMN.items():
            value = row.get(column_name)
            if pd.isna(value):
                continue
            period_values[return_period] = float(value)
        if period_values:
            lookup.setdefault(station_id, {})[display_model] = period_values
    return lookup


def _lookup_conchi_nrmse(
    lookup: dict[str, dict[str, dict[float, float]]],
    station_id: str,
    model_name: str,
    return_period: float,
) -> float | None:
    station_values = lookup.get(_normalize_station_id_for_conchi(station_id))
    if station_values is None:
        return None
    model_values = station_values.get(model_name)
    if model_values is None:
        return None
    value = model_values.get(float(return_period))
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _format_return_period_station_tick(
    station_id: str,
    station_names: dict[str, str] | None,
    *,
    show_only_station_id: bool = DEFAULT_RETURN_PERIOD_LINEPLOT_SHOW_ONLY_STATION_ID,
) -> str:
    if show_only_station_id:
        return f"Station {station_id}"
    if not station_names:
        return station_id
    station_name = station_names.get(station_id)
    if not station_name:
        return station_id
    return f"{station_id}: {str(station_name).strip().replace(' en ', ' in ')}"


def _ensure_return_period_model_color(gnn_model_label: str) -> None:
    if gnn_model_label not in RETURN_PERIOD_MODEL_COLORS:
        RETURN_PERIOD_MODEL_COLORS[gnn_model_label] = RETURN_PERIOD_MODEL_COLORS["GNN-LSTM"]


def _stations_with_conchi_nrmse_data(
    station_ids: list[str],
    conchi_lookup: dict[str, dict[str, dict[float, float]]],
) -> list[str]:
    return [
        station_id
        for station_id in station_ids
        if _normalize_station_id_for_conchi(station_id) in conchi_lookup
    ]


def _collect_return_period_nrmse_values(
    station_frames: dict[str, pd.DataFrame],
    station_ids: list[str],
    *,
    return_periods: list[float],
    return_periods_path: str | Path,
    test_start_date: pd.Timestamp | None,
    examine_train_test: bool = True,
    conchi_nrmse_path: str | Path,
    training_scenario: ConchiTrainingScenario,
    gnn_model_label: str = "GNN-LSTM",
    report_missing_points: bool = True,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> tuple[dict[float, dict[str, float]], dict[str, dict[str, dict[float, float]]]]:
    thresholds_lookup = _load_return_period_thresholds_lookup(return_periods_path)
    conchi_lookup = _load_conchi_nrmse_lookup(
        conchi_nrmse_path,
        training_scenario=training_scenario,
    )

    normalized_return_periods = [float(return_period) for return_period in return_periods]
    gnn_values: dict[float, dict[str, float]] = {}
    missing_reasons: dict[float, dict[str, str]] = {}
    for return_period in normalized_return_periods:
        gnn_values[return_period] = {}
        missing_reasons[return_period] = {}
        for station_id in station_ids:
            frame = station_frames.get(station_id)
            if frame is None or frame.empty:
                gnn_values[return_period][station_id] = float("nan")
                missing_reasons[return_period][station_id] = "missing prediction frame for station"
                continue
            station_thresholds = _get_station_return_period_thresholds(
                thresholds_lookup,
                station_id,
            )
            nrmse_value, missing_reason = _compute_gnn_nrmse_for_return_period(
                frame,
                return_period=return_period,
                analysis_return_periods=normalized_return_periods,
                station_thresholds=station_thresholds,
                test_start_date=test_start_date,
                examine_train_test=examine_train_test,
                observed_col=observed_col,
                predicted_col=predicted_col,
            )
            gnn_values[return_period][station_id] = nrmse_value
            if missing_reason is not None:
                missing_reasons[return_period][station_id] = missing_reason

    if report_missing_points:
        _report_missing_gnn_return_period_points(
            station_ids,
            normalized_return_periods,
            gnn_values,
            missing_reasons,
            gnn_model_label=gnn_model_label,
        )
    return gnn_values, conchi_lookup


def collect_return_period_nrmse_values(
    station_frames: dict[str, pd.DataFrame],
    station_ids: list[str],
    *,
    return_periods: list[float] | None = None,
    return_periods_path: str | Path = DEFAULT_RETURN_PERIODS_PATH,
    test_start_date: str | pd.Timestamp | None = None,
    examine_train_test: bool = True,
    conchi_nrmse_path: str | Path = DEFAULT_CONCHI_RETURN_PERIOD_NRMSE_PATH,
    training_scenario: ConchiTrainingScenario = "C2",
    gnn_model_label: str = "GNN-LSTM",
    report_missing_points: bool = True,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> tuple[dict[float, dict[str, float]], dict[str, dict[str, dict[float, float]]]]:
    if return_periods is None:
        return_periods = list(DEFAULT_RETURN_PERIODS)
    test_start = pd.Timestamp(test_start_date) if test_start_date is not None else None
    return _collect_return_period_nrmse_values(
        station_frames,
        station_ids,
        return_periods=return_periods,
        return_periods_path=return_periods_path,
        test_start_date=test_start,
        examine_train_test=examine_train_test,
        conchi_nrmse_path=conchi_nrmse_path,
        training_scenario=training_scenario,
        gnn_model_label=gnn_model_label,
        report_missing_points=report_missing_points,
        observed_col=observed_col,
        predicted_col=predicted_col,
    )


def _model_values_for_return_period(
    model_name: str,
    gnn_model_label: str,
    return_period: float,
    station_ids: list[str],
    gnn_values: dict[float, dict[str, float]],
    conchi_lookup: dict[str, dict[str, dict[float, float]]],
) -> list[float]:
    values: list[float] = []
    for station_id in station_ids:
        if model_name == gnn_model_label:
            value = _lookup_gnn_nrmse_value(gnn_values, return_period, station_id)
            if value is not None:
                values.append(value)
            continue
        conchi_value = _lookup_conchi_nrmse(
            conchi_lookup,
            station_id,
            model_name,
            return_period,
        )
        if conchi_value is not None:
            values.append(conchi_value)
    return values


def _return_period_model_legend_handles(
    gnn_model_label: str,
    *,
    include_mc_lstm: bool = True,
) -> list[plt.Artist]:
    model_order = _return_period_model_order(
        gnn_model_label,
        include_mc_lstm=include_mc_lstm,
    )
    return [
        plt.Line2D(
            [0],
            [0],
            color=RETURN_PERIOD_MODEL_COLORS[model_name],
            linewidth=2.5,
            label=model_name,
        )
        for model_name in model_order
    ]


def _draw_return_period_boxplot(
    ax: plt.Axes,
    *,
    return_periods: list[float],
    station_ids: list[str],
    gnn_model_label: str,
    gnn_values: dict[float, dict[str, float]],
    conchi_lookup: dict[str, dict[str, dict[float, float]]],
) -> list[plt.Artist]:
    model_order = [gnn_model_label, "LSTM", "MC-LSTM"]
    box_width = 0.18
    period_positions = np.arange(len(return_periods), dtype=np.float64)

    for period_idx, return_period in enumerate(return_periods):
        for model_idx, model_name in enumerate(model_order):
            values = _model_values_for_return_period(
                model_name,
                gnn_model_label,
                return_period,
                station_ids,
                gnn_values,
                conchi_lookup,
            )
            if not values:
                continue

            offset = (model_idx - (len(model_order) - 1) / 2.0) * box_width
            position = period_positions[period_idx] + offset
            boxplot = ax.boxplot(
                values,
                positions=[position],
                widths=box_width * 0.9,
                patch_artist=True,
                showfliers=True,
                whiskerprops={"linewidth": 1.0, "color": RETURN_PERIOD_MODEL_COLORS[model_name]},
                capprops={"linewidth": 1.0, "color": RETURN_PERIOD_MODEL_COLORS[model_name]},
                boxprops={"linewidth": 1.0, "color": RETURN_PERIOD_MODEL_COLORS[model_name]},
                medianprops={"linewidth": 1.5, "color": "#111111"},
                flierprops={
                    "marker": "o",
                    "markersize": 4,
                    "markerfacecolor": "white",
                    "markeredgecolor": RETURN_PERIOD_MODEL_COLORS[model_name],
                },
            )
            boxplot["boxes"][0].set_facecolor(RETURN_PERIOD_MODEL_COLORS[model_name])
            boxplot["boxes"][0].set_alpha(0.85)

        if period_idx < len(return_periods) - 1:
            divider_x = period_positions[period_idx] + 0.5
            ax.axvline(divider_x, color="#111111", linewidth=0.8, linestyle="-", alpha=0.25)

    ax.set_xticks(period_positions)
    ax.set_xticklabels([_format_return_period_label(period) for period in return_periods])
    ax.set_ylabel(r"$nRMSE$ (-)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    return _return_period_model_legend_handles(gnn_model_label)


def _compute_return_period_nrmse_ylim(
    station_ids: list[str],
    return_periods: list[float],
    *,
    gnn_model_label: str,
    gnn_values: dict[float, dict[str, float]],
    conchi_lookup: dict[str, dict[str, dict[float, float]]],
    include_mc_lstm: bool = True,
    padding: float = 0.05,
) -> tuple[float, float]:
    model_order = _return_period_model_order(
        gnn_model_label,
        include_mc_lstm=include_mc_lstm,
    )
    all_values: list[float] = []
    for station_id in station_ids:
        for return_period in return_periods:
            for model_name in model_order:
                if model_name == gnn_model_label:
                    value = _lookup_gnn_nrmse_value(gnn_values, return_period, station_id)
                else:
                    value = _lookup_conchi_nrmse(
                        conchi_lookup,
                        station_id,
                        model_name,
                        return_period,
                    )
                if value is not None and np.isfinite(value):
                    all_values.append(float(value))
    if not all_values:
        return (0.0, 1.0)
    ymax = float(max(all_values))
    margin = max(ymax * padding, 0.05)
    return (0.0, ymax + margin)


def _draw_return_period_lineplot(
    axes: np.ndarray,
    *,
    station_ids: list[str],
    return_periods: list[float],
    gnn_model_label: str,
    gnn_values: dict[float, dict[str, float]],
    conchi_lookup: dict[str, dict[str, dict[float, float]]],
    station_names: dict[str, str] | None = None,
    ylim: tuple[float, float] | None = None,
    grid_cols: int | None = None,
    fontsize: float = FONTSIZE_DEFAULT,
    include_mc_lstm: bool = False,
    show_only_station_id: bool = DEFAULT_RETURN_PERIOD_LINEPLOT_SHOW_ONLY_STATION_ID,
) -> list[plt.Artist]:
    model_order = _return_period_model_order(
        gnn_model_label,
        include_mc_lstm=include_mc_lstm,
    )
    x_positions = np.arange(len(return_periods), dtype=np.float64)
    x_labels = [_format_return_period_axis_label(period) for period in return_periods]
    axes_flat = np.atleast_1d(axes).ravel()

    for subplot_idx, station_id in enumerate(station_ids):
        ax = axes_flat[subplot_idx]
        for model_name in model_order:
            y_values: list[float] = []
            valid_x: list[float] = []
            for x_pos, return_period in zip(x_positions, return_periods):
                if model_name == gnn_model_label:
                    value = _lookup_gnn_nrmse_value(gnn_values, return_period, station_id)
                else:
                    value = _lookup_conchi_nrmse(
                        conchi_lookup,
                        station_id,
                        model_name,
                        return_period,
                    )
                if value is None or not np.isfinite(value):
                    continue
                valid_x.append(float(x_pos))
                y_values.append(float(value))

            if not valid_x:
                continue

            ax.plot(
                valid_x,
                y_values,
                color=RETURN_PERIOD_MODEL_COLORS[model_name],
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=model_name,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        if grid_cols is None or subplot_idx % grid_cols == 0:
            ax.set_ylabel(r"$nRMSE$ (-)", fontsize=fontsize)
        ax.set_ylim(ylim if ylim is not None else (0.0, None))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_title(
            _format_return_period_station_tick(
                station_id,
                station_names,
                show_only_station_id=show_only_station_id,
            ),
            fontsize=fontsize,
        )
        _apply_axis_font_size(ax, fontsize)

    if station_ids:
        if grid_cols is not None:
            grid_rows = math.ceil(len(station_ids) / grid_cols)
            bottom_row_start = (grid_rows - 1) * grid_cols
            for subplot_idx in range(len(station_ids)):
                if subplot_idx >= bottom_row_start:
                    axes_flat[subplot_idx].set_xlabel("Return period", fontsize=fontsize)
        else:
            axes_flat[len(station_ids) - 1].set_xlabel("Return period", fontsize=fontsize)
    return _return_period_model_legend_handles(
        gnn_model_label,
        include_mc_lstm=include_mc_lstm,
    )


def plot_return_period_nrmse_boxplots(
    station_frames: dict[str, pd.DataFrame],
    station_ids: list[str],
    *,
    gnn_model_label: str = "GNN-LSTM",
    return_periods: list[float] | None = None,
    return_periods_path: str | Path = DEFAULT_RETURN_PERIODS_PATH,
    test_start_date: str | pd.Timestamp | None = None,
    examine_train_test: bool = True,
    conchi_nrmse_path: str | Path = DEFAULT_CONCHI_RETURN_PERIOD_NRMSE_PATH,
    training_scenario: ConchiTrainingScenario = "C2",
    output_path: str | Path | None = None,
    show_plot: bool = False,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
) -> None:
    """
    Plot one boxplot figure where each return period groups nRMSE values across
    stations for GNN-LSTM, LSTM, and MC-LSTM.
    """
    if not station_ids:
        print("No stations provided for return-period nRMSE boxplots.")
        return

    if return_periods is None:
        return_periods = list(DEFAULT_RETURN_PERIODS)

    _ensure_return_period_model_color(gnn_model_label)
    test_start = pd.Timestamp(test_start_date) if test_start_date is not None else None
    gnn_values, conchi_lookup = _collect_return_period_nrmse_values(
        station_frames,
        station_ids,
        return_periods=return_periods,
        return_periods_path=return_periods_path,
        test_start_date=test_start,
        examine_train_test=examine_train_test,
        conchi_nrmse_path=conchi_nrmse_path,
        training_scenario=training_scenario,
        gnn_model_label=gnn_model_label,
        report_missing_points=False,
        observed_col=observed_col,
        predicted_col=predicted_col,
    )
    comparison_station_ids = _stations_with_conchi_nrmse_data(station_ids, conchi_lookup)
    if not comparison_station_ids:
        comparison_station_ids = station_ids
    if not comparison_station_ids:
        print("No stations with return-period nRMSE data for boxplots.")
        return

    fig, ax = plt.subplots(figsize=(max(1.4 * len(return_periods) * 2.0, 8.0), 5.0))
    legend_handles = _draw_return_period_boxplot(
        ax,
        return_periods=return_periods,
        station_ids=comparison_station_ids,
        gnn_model_label=gnn_model_label,
        gnn_values=gnn_values,
        conchi_lookup=conchi_lookup,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_return_period_nrmse_lineplots(
    station_frames: dict[str, pd.DataFrame],
    station_ids: list[str],
    *,
    gnn_model_label: str = "GNN-LSTM",
    return_periods: list[float] | None = None,
    return_periods_path: str | Path = DEFAULT_RETURN_PERIODS_PATH,
    test_start_date: str | pd.Timestamp | None = None,
    examine_train_test: bool = True,
    station_names: dict[str, str] | None = None,
    conchi_nrmse_path: str | Path = DEFAULT_CONCHI_RETURN_PERIOD_NRMSE_PATH,
    training_scenario: ConchiTrainingScenario = "C2",
    output_path: str | Path | None = None,
    individual_output_dir: str | Path | None = None,
    show_plot: bool = False,
    observed_col: str = "observed",
    predicted_col: str = "predicted",
    grid_rows: int = DEFAULT_RETURN_PERIOD_LINEPLOT_GRID_ROWS,
    grid_cols: int = DEFAULT_RETURN_PERIOD_LINEPLOT_GRID_COLS,
    show_only_station_id: bool = DEFAULT_RETURN_PERIOD_LINEPLOT_SHOW_ONLY_STATION_ID,
    separated_legend: bool = True,
    fontsize: float = FONTSIZE_DEFAULT,
    include_mc_lstm: bool = False,
) -> None:
    """
    Plot return-period nRMSE line charts in a station grid (2 rows x 2 columns by
    default). Each panel shows how nRMSE evolves across return periods for
    GNN-LSTM and LSTM (and optionally MC-LSTM). All panels share the same y-axis limits.

    When output_path is set, individual station PNGs are also saved under
    ``lineplot_return_periods/`` next to the combined figure unless
    individual_output_dir is provided explicitly.
    """
    if not station_ids:
        print("No stations provided for return-period nRMSE lineplots.")
        return

    if return_periods is None:
        return_periods = list(DEFAULT_RETURN_PERIODS)

    _ensure_return_period_model_color(gnn_model_label)
    test_start = pd.Timestamp(test_start_date) if test_start_date is not None else None
    gnn_values, conchi_lookup = _collect_return_period_nrmse_values(
        station_frames,
        station_ids,
        return_periods=return_periods,
        return_periods_path=return_periods_path,
        test_start_date=test_start,
        examine_train_test=examine_train_test,
        conchi_nrmse_path=conchi_nrmse_path,
        training_scenario=training_scenario,
        gnn_model_label=gnn_model_label,
        report_missing_points=True,
        observed_col=observed_col,
        predicted_col=predicted_col,
    )
    lineplot_station_ids = list(station_ids)
    normalized_return_periods = [float(return_period) for return_period in return_periods]
    shared_ylim = _compute_return_period_nrmse_ylim(
        lineplot_station_ids,
        normalized_return_periods,
        gnn_model_label=gnn_model_label,
        gnn_values=gnn_values,
        conchi_lookup=conchi_lookup,
        include_mc_lstm=include_mc_lstm,
    )
    legend_handles = _return_period_model_legend_handles(
        gnn_model_label,
        include_mc_lstm=include_mc_lstm,
    )

    n_stations = len(lineplot_station_ids)
    effective_grid_rows = max(1, grid_rows)
    effective_grid_cols = max(1, grid_cols, math.ceil(n_stations / effective_grid_rows))
    fig_width = max(4.2 * effective_grid_cols, 8.5)
    fig_height = max(2.8 * effective_grid_rows, 4.0)
    fig, axes = plt.subplots(
        effective_grid_rows,
        effective_grid_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    _draw_return_period_lineplot(
        axes,
        station_ids=lineplot_station_ids,
        return_periods=normalized_return_periods,
        gnn_model_label=gnn_model_label,
        gnn_values=gnn_values,
        conchi_lookup=conchi_lookup,
        station_names=station_names,
        ylim=shared_ylim,
        grid_cols=effective_grid_cols,
        fontsize=fontsize,
        include_mc_lstm=include_mc_lstm,
        show_only_station_id=show_only_station_id,
    )
    for ax in axes.ravel()[n_stations:]:
        ax.set_visible(False)
    if separated_legend:
        if output_path is not None:
            combined_legend_path = (
                Path(output_path).parent / f"{Path(output_path).stem}_legend.png"
            )
            _save_separated_figure_legend(
                legend_handles,
                combined_legend_path,
                fontsize=fontsize,
                ncol=len(legend_handles),
            )
    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=fontsize,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if output_path is not None or individual_output_dir is not None:
        if individual_output_dir is None:
            if output_path is None:
                return
            individual_output_dir = (
                Path(output_path).parent / DEFAULT_RETURN_PERIOD_LINEPLOT_INDIVIDUAL_DIRNAME
            )
        individual_output_dir = Path(individual_output_dir)
        individual_output_dir.mkdir(parents=True, exist_ok=True)
        if separated_legend:
            _save_separated_figure_legend(
                legend_handles,
                individual_output_dir / "return_period_nrmse_legend.png",
                fontsize=fontsize,
                ncol=len(legend_handles),
            )
        for station_id in lineplot_station_ids:
            fig_single, ax_single = plt.subplots(figsize=(6.0, 4.0))
            _draw_return_period_lineplot(
                np.array([ax_single]),
                station_ids=[station_id],
                return_periods=normalized_return_periods,
                gnn_model_label=gnn_model_label,
                gnn_values=gnn_values,
                conchi_lookup=conchi_lookup,
                station_names=station_names,
                ylim=shared_ylim,
                grid_cols=None,
                fontsize=fontsize,
                include_mc_lstm=include_mc_lstm,
                show_only_station_id=show_only_station_id,
            )
            if not separated_legend:
                ax_single.legend(loc="best", frameon=False, fontsize=fontsize)
            fig_single.tight_layout()
            safe_station_id = str(station_id).replace("/", "_").replace("\\", "_")
            fig_single.savefig(
                individual_output_dir / f"{safe_station_id}_return_period_nrmse.png",
                dpi=150,
                bbox_inches="tight",
            )
            if show_plot:
                plt.show()
            else:
                plt.close(fig_single)


def _finite_station_metric_values(values_by_station: dict[str, float]) -> list[float]:
    return [
        float(value)
        for value in values_by_station.values()
        if value is not None and np.isfinite(value)
    ]


def compute_station_metric_boxplot_ylim(
    metric_by_param_label_groups: list[dict[str, dict[str, float]]],
    *,
    padding: float = 0.05,
    ymin: float | None = None,
) -> tuple[float, float]:
    all_values: list[float] = []
    for group in metric_by_param_label_groups:
        for values_by_station in group.values():
            all_values.extend(_finite_station_metric_values(values_by_station))
    if not all_values:
        return (ymin if ymin is not None else 0.0, 1.0)

    data_ymin = float(min(all_values))
    ymax = float(max(all_values))
    if ymin is not None:
        lower = ymin
        reference_ymin = ymin
    else:
        span = ymax - data_ymin
        lower_margin = max(abs(ymax) * padding, 0.05) if span <= 0 else span * padding
        lower = data_ymin - lower_margin
        reference_ymin = data_ymin

    span = ymax - reference_ymin
    upper_margin = max(abs(ymax) * padding, 0.05) if span <= 0 else span * padding
    return (lower, ymax + upper_margin)


def _format_station_metric_param_tick_label(
    label: str,
    varied_param: str,
    *,
    format_labels: bool,
) -> str:
    if varied_param == "weighted_adj":
        return WEIGHTED_ADJ_BOXPLOT_LABELS.get(label, label)
    if format_labels:
        return label.replace("_", " ")
    return label


def _draw_station_metric_boxplot_on_axis(
    ax: plt.Axes,
    metric_by_param_label: dict[str, dict[str, float]],
    *,
    error_metric: str,
    varied_param: str,
    baseline_label: str | None = None,
    highlight_baseline: bool = True,
    baseline_color: str = "#d95f02",
    default_color: str = "#1f8a8a",
    ylim: tuple[float, float] | None = None,
    show_ylabel: bool = True,
    format_labels: bool = True,
    fontsize: float = FONTSIZE_DEFAULT,
    show_outliers_as_dots: bool = False,
    show_title: bool = True,
) -> bool:
    if not metric_by_param_label:
        return False

    display_varied_param = varied_param.replace("_", " ") if format_labels else varied_param
    labels = list(metric_by_param_label.keys())
    display_labels = [
        _format_station_metric_param_tick_label(
            label,
            varied_param,
            format_labels=format_labels,
        )
        for label in labels
    ]
    data = [_finite_station_metric_values(metric_by_param_label[label]) for label in labels]
    if not any(data):
        return False

    boxplot = ax.boxplot(
        data,
        tick_labels=display_labels,
        showfliers=show_outliers_as_dots,
        whis=(0, 100) if not show_outliers_as_dots else 1.5,
        patch_artist=True,
    )
    for idx, box in enumerate(boxplot["boxes"]):
        if hasattr(box, "set_facecolor"):
            use_baseline_color = (
                highlight_baseline
                and baseline_label is not None
                and labels[idx] == baseline_label
            )
            box.set_facecolor(baseline_color if use_baseline_color else default_color)
            box.set_alpha(0.85)
    ax.set_xlabel(display_varied_param, fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel(error_metric, fontsize=fontsize)
    if show_title:
        if format_labels:
            ax.set_title(f"{error_metric} by {display_varied_param}", fontsize=fontsize)
        else:
            ax.set_title(f"{error_metric} across stations by {varied_param}", fontsize=fontsize)
    if varied_param == "weighted_adj":
        ax.tick_params(axis="x", labelrotation=45)
        for tick_label in ax.get_xticklabels():
            tick_label.set_ha("right")
    if ylim is not None:
        ax.set_ylim(ylim)
    _apply_axis_font_size(ax, fontsize)
    return True


def plot_station_metric_boxplot_by_param_values(
    metric_by_param_label: dict[str, dict[str, float]],
    *,
    error_metric: str = "NSE",
    varied_param: str = "parameter",
    output_path: str | Path | None = None,
    show_plot: bool = False,
    baseline_label: str | None = None,
    highlight_baseline: bool = True,
    baseline_color: str = "#d95f02",
    default_color: str = "#1f8a8a",
    ylim: tuple[float, float] | None = None,
    format_labels: bool = True,
    fontsize: float = FONTSIZE_DEFAULT,
    show_outliers_as_dots: bool = False,
) -> None:
    """
    Draw one boxplot per parameter value showing the chosen metric across stations.

    metric_by_param_label maps a human-readable parameter value label to
    station_id -> metric value.
    """
    if not metric_by_param_label:
        print("No parameter values provided for comparison boxplot.")
        return

    labels = list(metric_by_param_label.keys())
    data = [_finite_station_metric_values(metric_by_param_label[label]) for label in labels]
    if not any(data):
        print(f"No finite {error_metric} values available for comparison boxplot.")
        return

    fig, ax = plt.subplots(figsize=(max(1.2 * len(labels), 6.0), 5.0))
    _draw_station_metric_boxplot_on_axis(
        ax,
        metric_by_param_label,
        error_metric=error_metric,
        varied_param=varied_param,
        baseline_label=baseline_label,
        highlight_baseline=highlight_baseline,
        baseline_color=baseline_color,
        default_color=default_color,
        ylim=ylim,
        show_ylabel=True,
        format_labels=format_labels,
        fontsize=fontsize,
        show_outliers_as_dots=show_outliers_as_dots,
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_station_metric_boxplot_grid(
    metric_by_varied_param: dict[str, dict[str, dict[str, float]]],
    *,
    error_metric: str = "NSE",
    baseline_labels: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    show_plot: bool = False,
    highlight_baseline: bool = True,
    baseline_color: str = "#d95f02",
    default_color: str = "#1f8a8a",
    ylim: tuple[float, float] | None = None,
    format_labels: bool = True,
    rows: int = 4,
    fontsize: float = FONTSIZE_DEFAULT,
    show_outliers_as_dots: bool = False,
    show_titles: bool = False,
) -> None:
    """Draw a grid of boxplots, one panel per varied parameter, sharing a y-axis."""
    varied_params = [param for param, values in metric_by_varied_param.items() if values]
    if not varied_params:
        print("No parameter groups provided for comparison boxplot grid.")
        return

    if ylim is None:
        ylim = compute_station_metric_boxplot_ylim(
            [metric_by_varied_param[param] for param in varied_params]
        )

    n_params = len(varied_params)
    effective_rows = max(1, rows)
    ncol = math.ceil(n_params / effective_rows)
    fig, axes = plt.subplots(
        effective_rows,
        ncol,
        figsize=(max(3.5 * ncol, 8.0), max(2.5 * effective_rows, 5.0)),
        sharey=True,
        squeeze=False,
    )
    baseline_labels = baseline_labels or {}
    plotted_any = False
    for param_idx, varied_param in enumerate(varied_params):
        row_idx = param_idx // ncol
        col_idx = param_idx % ncol
        ax = axes[row_idx, col_idx]
        plotted = _draw_station_metric_boxplot_on_axis(
            ax,
            metric_by_varied_param[varied_param],
            error_metric=error_metric,
            varied_param=varied_param,
            baseline_label=baseline_labels.get(varied_param),
            highlight_baseline=highlight_baseline,
            baseline_color=baseline_color,
            default_color=default_color,
            ylim=ylim,
            show_ylabel=(col_idx == 0),
            format_labels=format_labels,
            fontsize=fontsize,
            show_outliers_as_dots=show_outliers_as_dots,
            show_title=show_titles,
        )
        plotted_any = plotted_any or plotted
        if not plotted:
            ax.set_visible(False)

    for extra_idx in range(n_params, effective_rows * ncol):
        extra_row = extra_idx // ncol
        extra_col = extra_idx % ncol
        axes[extra_row, extra_col].set_visible(False)

    if not plotted_any:
        print(f"No finite {error_metric} values available for comparison boxplot grid.")
        plt.close(fig)
        return

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


### TODO END REVISAR CHAT









def plot_graph_mse_map(
    weighted_adj_matrix: pd.DataFrame,
    static_info_path: str | Path,
    mse_by_station: dict[str, float],
    output_html: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    plot_graph_error_map(
        weighted_adj_matrix,
        static_info_path,
        mse_by_station,
        error_metric="MSE",
        output_html=output_html,
        show_plot=show_plot,
    )
