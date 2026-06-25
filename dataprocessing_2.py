#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adj_matrix_visualize_maps_GNNs import DEFAULT_STATIC_INFO_PATH
from data_processing import _align_dataframe, _streamflow_column
from visuals import load_station_name_map

# DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_allstations_plus_static_checked_v2.pkl"
DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals"
DEFAULT_OUTPUT_SUBDIR = "basin_summary_graphics"
DEFAULT_TRAIN_FRACTION = 0.8
ONLY_3_MONTHS = True
SAME_Y_AXIS = False
NORMALIZE = True
SHOW_GRID = False
INCLUDE_ATMOSPHERIC_INDEXES = False
FONT_SIZE = 15
MEAN_CYCLE_FONT_SIZE = FONT_SIZE + 2
PUT_ONLY_FIRST_Y_TICKS = True
DEFAULT_STATION_SUMMARY_DECIMALS = 2

GRID_TEMPERATURE_YLABEL = "Max temperature [°C]\nMin temperature [°C]"
GRID_TEMPERATURE_YLABEL_NORMALIZED = "Max temperature (z-score)\nMin temperature (z-score)"
GRID_STREAMFLOW_YLABEL_NORMALIZED = "Normalized \n streamflow [-]"

STATION_SUMMARY_NUMERIC_COLUMNS = (
    "Drainage Area (km2)",
    "Elevation gauging station (m.a.s.l.)",
    "Mean Q (m3/s)",
    "Q10 (m3/s)",
    "Q90 (m3/s)",
)

MEAN_LINE_COLOR = "#0b3d91"
MEAN_FILL_COLOR = "#4a7bc8"
TMAX_LINE_COLOR = "#c0392b"
TMAX_FILL_COLOR = "#e57373"
TMIN_LINE_COLOR = "#0b3d91"
TMIN_FILL_COLOR = "#4a7bc8"
PLOT_LINE_WIDTH = 1.25
PLOT_DPI = 150

MONTH_START_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
THREE_MONTH_TICK_INDICES = (0, 6, 11)

COLUMN_ALIASES: dict[str, list[str]] = {
    "pr": ["pr", "Catchment-averaged precipitation [mm]"],
    "Humidity": ["Humidity", "Air humidity [%]"],
    "tmax_total": ["tmax_total", "Maximum temperature [°C]"],
    "tmin_total": ["tmin_total", "Minimum temperature [°C]"],
    "SPEI": ["SPEI", "SPEI [−]"],
    "nao": ["nao", "NAO [−]"],
    "WEMO": ["WEMO", "WeMO [−]"],
}

STATIC_COLUMNS = {
    "catchment_area": "Catchment Area (km2)",
    "elevation": "Elevation gauging station (m.a.s.l.)",
    "agricultural": "Agricultural areas",
    "forests": "Forests",
    "shrub": "Shrub and/or herbaceous vegetation",
}

GNN_LSTM_VARIABLE_ROWS = [
    ("Cumulative precipitation", "I", "Climatic", "Daily", "AEMET"),
    ("Maximum temperature", "I", "Climatic", "Daily", "AEMET"),
    ("Minimum temperature", "I", "Climatic", "Daily", "AEMET"),
    ("Air humidity", "I", "Climatic", "Daily", "AEMET"),
    ("SPEI", "I", "Atmospheric", "Daily", "AEMET"),
    ("NAO", "I", "Atmospheric", "Daily", "WMO"),
    ("WEMO", "I", "Atmospheric", "Daily", "-"),
    ("Drainage area", "I", "Static", "-", "CHE"),
    ("Mean elevation", "I", "Static", "-", "CHE"),
    ("Forest fraction", "I", "Static", "-", "CLC"),
    ("Agricultural fraction", "I", "Static", "-", "CLC"),
    ("Herbaceous vegetation", "I", "Static", "-", "CLC"),
    ("Latitude", "I", "Static", "-", "CHE"),
    ("Longitude", "I", "Static", "-", "CHE"),
    ("Streamflow", "O", "Hydrological", "Daily", "CHE/CEDEX"),
]


@dataclass(frozen=True)
class VariablePlotSpec:
    key: str
    label: str
    y_label: str
    unit: str
    column_candidates: tuple[str, ...]
    clamp_below_zero: bool = True
    line_color: str = MEAN_LINE_COLOR
    fill_color: str = MEAN_FILL_COLOR
    is_temperature: bool = False


SINGLE_VARIABLE_SPECS: tuple[VariablePlotSpec, ...] = (
    VariablePlotSpec(
        key="streamflow",
        label="streamflow",
        y_label="Streamflow [m³/s]",
        unit="m³/s",
        column_candidates=(),
        clamp_below_zero=True,
    ),
    VariablePlotSpec(
        key="precipitation",
        label="precipitation",
        y_label="Precipitation [mm]",
        unit="mm",
        column_candidates=("pr",),
        clamp_below_zero=True,
    ),
    VariablePlotSpec(
        key="humidity",
        label="humidity",
        y_label="Humidity [%]",
        unit="%",
        column_candidates=("Humidity",),
        clamp_below_zero=True,
    ),
)

TEMPERATURE_SPEC = VariablePlotSpec(
    key="temperature",
    label="temperature",
    y_label="Temperature [°C]",
    unit="°C",
    column_candidates=("tmax_total", "tmin_total"),
    clamp_below_zero=False,
    is_temperature=True,
)

ATMOSPHERIC_INDEX_SPECS: tuple[VariablePlotSpec, ...] = (
    VariablePlotSpec(
        key="spei",
        label="SPEI",
        y_label="SPEI [−]",
        unit="−",
        column_candidates=("SPEI",),
        clamp_below_zero=False,
    ),
    VariablePlotSpec(
        key="nao",
        label="NAO",
        y_label="NAO [−]",
        unit="−",
        column_candidates=("nao",),
        clamp_below_zero=False,
    ),
    VariablePlotSpec(
        key="wemo",
        label="WEMO",
        y_label="WEMO [−]",
        unit="−",
        column_candidates=("WEMO",),
        clamp_below_zero=False,
    ),
)


def build_plot_specs(*, include_atmospheric_indexes: bool = INCLUDE_ATMOSPHERIC_INDEXES) -> tuple[VariablePlotSpec, ...]:
    specs: list[VariablePlotSpec] = list(SINGLE_VARIABLE_SPECS) + [TEMPERATURE_SPEC]
    if include_atmospheric_indexes:
        specs.extend(ATMOSPHERIC_INDEX_SPECS)
    return tuple(specs)


def _iter_station_frames(data: dict) -> Iterable[tuple[str, pd.DataFrame]]:
    for station_id, frame in data.items():
        if isinstance(frame, pd.DataFrame):
            yield str(station_id), frame


def load_pickle_stations(pickle_path: str | Path) -> dict[str, pd.DataFrame]:
    pickle_path = Path(pickle_path)
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Pickle file {pickle_path} must contain dict[str, DataFrame], got {type(data)}")
    frames = {station_id: _align_dataframe(df) for station_id, df in _iter_station_frames(data)}
    if not frames:
        raise ValueError(f"No station DataFrames found in {pickle_path}")
    return frames


def load_static_info(static_info_path: str | Path) -> pd.DataFrame:
    static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
    if "station_id" not in static_info_df.columns:
        static_info_df = static_info_df.reset_index()
    return static_info_df.set_index("station_id")


def resolve_column_name(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        aliases = COLUMN_ALIASES.get(candidate, [candidate])
        for alias in aliases:
            if alias in df.columns:
                return alias
    raise ValueError(f"None of {candidates} found in DataFrame columns: {list(df.columns)}")


def resolve_streamflow_column(df: pd.DataFrame) -> str:
    return _streamflow_column(df)


def common_date_index(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    common_index: pd.DatetimeIndex | None = None
    for df in frames.values():
        station_index = pd.DatetimeIndex(df.index)
        common_index = station_index if common_index is None else common_index.intersection(station_index)
    if common_index is None or common_index.empty:
        raise ValueError("No common dates across stations")
    return common_index.sort_values()


def format_station_id(station_id: str) -> str:
    normalized = str(station_id).strip().upper()
    if normalized.startswith("A"):
        normalized = normalized[1:]
    return f"A{normalized.zfill(3)}"


def _format_station_display_name(full_name: str) -> str:
    return full_name.strip().replace(" en ", " in ")


def _short_station_name(full_name: str) -> str:
    marker = " en "
    if marker in full_name:
        return full_name.split(marker, 1)[1].strip()
    return full_name.strip()


def format_station_label(station_id: str, station_names: dict[str, str] | None) -> str:
    station_code = format_station_id(station_id)
    if not station_names:
        return station_code
    name = station_names.get(station_id)
    if not name:
        return station_code
    return f"{station_code}: {_format_station_display_name(name)}"


def _mean_cycle_station_label(station_label: str) -> str:
    return station_label.replace(" en ", " in ")


def _mean_cycle_y_label(spec: VariablePlotSpec, *, normalized: bool) -> str:
    if spec.key == "streamflow" and not normalized:
        return "Mean streamflow"
    return _variable_y_label(spec, normalized=normalized)


def _apply_axis_font_size(axis: plt.Axes, font_size: float) -> None:
    axis.xaxis.label.set_size(font_size)
    axis.yaxis.label.set_size(font_size)
    axis.tick_params(labelsize=font_size)
    title = axis.get_title()
    if title:
        axis.set_title(title, fontsize=font_size)


def _plot_font_rc_context(font_size: float):
    return plt.rc_context(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def format_mean_subtitle(mean_value: float, unit: str, *, streamflow: bool = False) -> str:
    if streamflow:
        if not np.isfinite(mean_value):
            return f"Mean flow: N/A [{unit}]"
        return f"Mean flow: {mean_value:.2f} [{unit}]"
    if not np.isfinite(mean_value):
        return f"Mean: N/A [{unit}]"
    return f"Mean: {mean_value:.2f} [{unit}]"


def decimal_to_dms(value: float, *, is_latitude: bool) -> str:
    if not np.isfinite(value):
        return ""
    hemisphere = "N" if is_latitude else "E"
    if value < 0:
        hemisphere = "S" if is_latitude else "W"
        value = abs(value)
    degrees = int(value)
    minutes_float = (value - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60.0
    return f"{degrees}°{minutes:02d}'{seconds:04.1f}\"{hemisphere}"


def summarize_station_date_ranges(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for station_id, df in sorted(frames.items()):
        if df.empty:
            continue
        rows.append(
            {
                "station_id": station_id,
                "start_date": str(df.index.min().date()),
                "end_date": str(df.index.max().date()),
                "num_days": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def compute_train_test_ranges(
    common_index: pd.DatetimeIndex,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
) -> dict[str, pd.Timestamp | int | float]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    num_days = len(common_index)
    split_idx = int(num_days * train_fraction)
    if split_idx <= 0 or split_idx >= num_days:
        raise ValueError(
            f"Invalid split for {num_days} common days and train_fraction={train_fraction}"
        )

    train_index = common_index[:split_idx]
    test_index = common_index[split_idx:]
    return {
        "num_days": num_days,
        "train_fraction": train_fraction,
        "test_fraction": 1.0 - train_fraction,
        "split_idx": split_idx,
        "train_start": train_index[0],
        "train_end": train_index[-1],
        "train_days": len(train_index),
        "test_start": test_index[0],
        "test_end": test_index[-1],
        "test_days": len(test_index),
    }


def print_date_summary(
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    split_info: dict[str, pd.Timestamp | int | float],
) -> None:
    per_station = summarize_station_date_ranges(frames)
    print("=" * 72)
    print("Pickle date coverage")
    print("=" * 72)
    print(f"Stations: {len(frames)}")
    print(
        f"Common date range across all stations: "
        f"{common_index.min().date()} to {common_index.max().date()} "
        f"({len(common_index)} days)"
    )
    print("\nPer-station ranges:")
    for _, row in per_station.iterrows():
        print(
            f"  {row['station_id']}: {row['start_date']} -> {row['end_date']} "
            f"({row['num_days']} days)"
        )

    print("\nChronological train/test split (common timeline):")
    print(
        f"  Train fraction: {split_info['train_fraction']:.2f} "
        f"({split_info['train_days']} days)"
    )
    print(
        f"  Test fraction:  {split_info['test_fraction']:.2f} "
        f"({split_info['test_days']} days)"
    )
    print(
        f"  Train days: {split_info['train_start'].date()} -> "
        f"{split_info['train_end'].date()}"
    )
    print(
        f"  Test days:  {split_info['test_start'].date()} -> "
        f"{split_info['test_end'].date()}"
    )
    print("=" * 72)


def normalize_series(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series
    mean = float(valid.mean())
    std = float(valid.std())
    if not np.isfinite(std) or std == 0.0:
        return series - mean
    return (series - mean) / std


def series_mean(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def static_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if values.size == 0:
        return float("nan")
    return float(values[0])


def compute_mean_by_doy(
    series: pd.Series,
    *,
    clamp_below_zero: bool = True,
) -> pd.DataFrame:
    values = series.dropna()
    if clamp_below_zero:
        values = values.clip(lower=0.0)
    frame = values.to_frame(name="value")
    frame["day_of_year"] = frame.index.dayofyear
    stats = frame.groupby("day_of_year", sort=True)["value"].agg(["mean", "std"])
    stats["std"] = stats["std"].fillna(0.0)
    if clamp_below_zero:
        stats["mean"] = stats["mean"].clip(lower=0.0)
        stats["std"] = stats["std"].clip(lower=0.0)
    return stats


def _month_axis_ticks(only_3_months: bool) -> tuple[list[int], list[str]]:
    if only_3_months:
        indices = THREE_MONTH_TICK_INDICES
    else:
        indices = range(len(MONTH_LABELS))
    return [MONTH_START_DOY[index] for index in indices], [MONTH_LABELS[index] for index in indices]


def _plot_value_arrays(
    stats: pd.DataFrame,
    *,
    clamp_below_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = stats.index.to_numpy()
    mean_values = stats["mean"].to_numpy(dtype=float)
    std_values = stats["std"].to_numpy(dtype=float)
    if clamp_below_zero:
        mean_values = np.clip(mean_values, 0.0, None)
        std_values = np.clip(std_values, 0.0, None)
        lower_band = np.maximum(mean_values - std_values, 0.0)
    else:
        lower_band = mean_values - std_values
    upper_band = mean_values + std_values
    return x, mean_values, lower_band, upper_band


def _plot_y_max(stats: pd.DataFrame, *, clamp_below_zero: bool = True) -> float:
    _, _, _, upper_band = _plot_value_arrays(stats, clamp_below_zero=clamp_below_zero)
    if upper_band.size == 0:
        return 0.0
    return float(np.max(upper_band))


def _plot_y_min(stats: pd.DataFrame, *, clamp_below_zero: bool = True) -> float:
    _, _, lower_band, _ = _plot_value_arrays(stats, clamp_below_zero=clamp_below_zero)
    if lower_band.size == 0:
        return 0.0
    return float(np.min(lower_band))


def _shared_y_limits(
    station_stats: dict[str, pd.DataFrame],
    *,
    clamp_below_zero: bool = True,
) -> tuple[float, float]:
    if not station_stats:
        return 0.0, 0.0
    y_min = min(_plot_y_min(stats, clamp_below_zero=clamp_below_zero) for stats in station_stats.values())
    y_max = max(_plot_y_max(stats, clamp_below_zero=clamp_below_zero) for stats in station_stats.values())
    return y_min, y_max


def _temperature_y_limits(
    tmax_stats: dict[str, pd.DataFrame],
    tmin_stats: dict[str, pd.DataFrame],
) -> tuple[float, float]:
    y_min = min(_plot_y_min(stats, clamp_below_zero=False) for stats in tmin_stats.values())
    y_max = max(_plot_y_max(stats, clamp_below_zero=False) for stats in tmax_stats.values())
    return y_min, y_max


def _apply_y_limits(
    axis: plt.Axes,
    *,
    clamp_below_zero: bool,
    y_min: float | None,
    y_max: float | None,
) -> None:
    if y_min is not None and y_max is not None:
        axis.set_ylim(y_min, y_max)
    elif y_max is not None:
        axis.set_ylim(0, max(y_max, 0.0))
    elif clamp_below_zero:
        axis.set_ylim(bottom=0)


def _style_cycle_axis(
    axis: plt.Axes,
    *,
    only_3_months: bool,
    show_xlabel: bool,
    show_ylabel: bool,
    y_label: str,
    show_grid: bool,
    x_label: str | None = "Day of year",
) -> None:
    month_ticks, month_labels = _month_axis_ticks(only_3_months)
    axis.set_xlim(1, 366)
    axis.set_xticks(month_ticks)
    if show_xlabel:
        axis.set_xticklabels(month_labels)
        if x_label is not None:
            axis.set_xlabel(x_label)
    else:
        axis.set_xticklabels([])
    if show_ylabel:
        axis.set_ylabel(y_label)
    if show_grid:
        axis.grid(True, alpha=0.3)


def plot_mean_cycle(
    stats: pd.DataFrame,
    spec: VariablePlotSpec,
    *,
    station_label: str,
    mean_value: float,
    output_path: Path | None,
    show_plot: bool,
    only_3_months: bool = ONLY_3_MONTHS,
    clamp_below_zero: bool = True,
    y_min: float | None = None,
    y_max: float | None = None,
    normalized: bool = False,
    show_grid: bool = SHOW_GRID,
    axis: plt.Axes | None = None,
    show_title: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    font_size: float = MEAN_CYCLE_FONT_SIZE,
) -> plt.Axes:
    x, mean_values, lower_band, upper_band = _plot_value_arrays(
        stats,
        clamp_below_zero=clamp_below_zero,
    )
    y_label = _mean_cycle_y_label(spec, normalized=normalized)
    title_suffix = f"normalized {spec.label} cycle" if normalized else f"{spec.label} cycle"
    streamflow = spec.key == "streamflow"
    display_label = _mean_cycle_station_label(station_label)

    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(12, 4))
        own_figure = True
    else:
        fig = axis.figure
        own_figure = False

    axis.fill_between(
        x,
        lower_band,
        upper_band,
        color=spec.fill_color,
        alpha=0.35,
        linewidth=0,
    )
    axis.plot(
        x,
        mean_values,
        color=spec.line_color,
        linewidth=PLOT_LINE_WIDTH,
    )
    if show_title:
        subtitle = format_mean_subtitle(mean_value, spec.unit, streamflow=streamflow)
        if streamflow:
            axis.set_title(f"{display_label}\n{subtitle}")
        else:
            axis.set_title(f"Mean daily {title_suffix} — {display_label}\n{subtitle}")
    _style_cycle_axis(
        axis,
        only_3_months=only_3_months,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        y_label=y_label,
        show_grid=show_grid,
        x_label=None,
    )
    _apply_y_limits(axis, clamp_below_zero=clamp_below_zero, y_min=y_min, y_max=y_max)
    _apply_axis_font_size(axis, font_size)

    if own_figure:
        fig.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    return axis


def plot_temperature_cycle(
    tmax_stats: pd.DataFrame,
    tmin_stats: pd.DataFrame,
    *,
    station_label: str,
    mean_tmax: float,
    mean_tmin: float,
    output_path: Path | None,
    show_plot: bool,
    only_3_months: bool = ONLY_3_MONTHS,
    clamp_below_zero: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
    normalized: bool = False,
    show_grid: bool = SHOW_GRID,
    axis: plt.Axes | None = None,
    show_title: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> plt.Axes:
    x_max, max_mean, max_lower, max_upper = _plot_value_arrays(
        tmax_stats,
        clamp_below_zero=clamp_below_zero,
    )
    x_min, min_mean, min_lower, min_upper = _plot_value_arrays(
        tmin_stats,
        clamp_below_zero=clamp_below_zero,
    )
    y_label = "Normalized temperature (z-score)" if normalized else TEMPERATURE_SPEC.y_label
    title_suffix = "normalized temperature cycle" if normalized else "temperature cycle"

    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(12, 4))
        own_figure = True
    else:
        fig = axis.figure
        own_figure = False

    axis.fill_between(x_max, max_lower, max_upper, color=TMAX_FILL_COLOR, alpha=0.25, linewidth=0)
    axis.fill_between(x_min, min_lower, min_upper, color=TMIN_FILL_COLOR, alpha=0.25, linewidth=0)
    axis.plot(x_max, max_mean, color=TMAX_LINE_COLOR, linewidth=PLOT_LINE_WIDTH, label="Mean tmax")
    axis.plot(x_min, min_mean, color=TMIN_LINE_COLOR, linewidth=PLOT_LINE_WIDTH, label="Mean tmin")
    if show_title:
        axis.set_title(
            f"Mean daily {title_suffix} — {station_label}\n"
            f"Mean tmax: {mean_tmax:.2f} [{TEMPERATURE_SPEC.unit}] | "
            f"Mean tmin: {mean_tmin:.2f} [{TEMPERATURE_SPEC.unit}]"
        )
    _style_cycle_axis(
        axis,
        only_3_months=only_3_months,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        y_label=y_label,
        show_grid=show_grid,
    )
    _apply_y_limits(axis, clamp_below_zero=clamp_below_zero, y_min=y_min, y_max=y_max)
    if show_title and show_legend:
        axis.legend(loc="upper right")

    if own_figure:
        fig.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    return axis


def _resolve_series_for_spec(
    frame: pd.DataFrame,
    common_index: pd.DatetimeIndex,
    spec: VariablePlotSpec,
) -> pd.Series:
    if spec.key == "streamflow":
        column = resolve_streamflow_column(frame)
    else:
        column = resolve_column_name(frame, spec.column_candidates)
    return frame.loc[common_index, column]


def _build_single_variable_stats(
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    spec: VariablePlotSpec,
    *,
    normalize: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    station_stats: dict[str, pd.DataFrame] = {}
    station_means: dict[str, float] = {}
    for station_id, frame in sorted(frames.items()):
        aligned = _resolve_series_for_spec(frame, common_index, spec)
        station_means[station_id] = series_mean(aligned)
        values = normalize_series(aligned) if normalize else aligned
        station_stats[station_id] = compute_mean_by_doy(
            values,
            clamp_below_zero=spec.clamp_below_zero and not normalize,
        )
    return station_stats, station_means


def _build_temperature_stats(
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    *,
    normalize: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, float], dict[str, float]]:
    tmax_stats: dict[str, pd.DataFrame] = {}
    tmin_stats: dict[str, pd.DataFrame] = {}
    mean_tmax: dict[str, float] = {}
    mean_tmin: dict[str, float] = {}
    for station_id, frame in sorted(frames.items()):
        tmax_col = resolve_column_name(frame, ("tmax_total",))
        tmin_col = resolve_column_name(frame, ("tmin_total",))
        aligned_tmax = frame.loc[common_index, tmax_col]
        aligned_tmin = frame.loc[common_index, tmin_col]
        mean_tmax[station_id] = series_mean(aligned_tmax)
        mean_tmin[station_id] = series_mean(aligned_tmin)
        tmax_values = normalize_series(aligned_tmax) if normalize else aligned_tmax
        tmin_values = normalize_series(aligned_tmin) if normalize else aligned_tmin
        tmax_stats[station_id] = compute_mean_by_doy(tmax_values, clamp_below_zero=False)
        tmin_stats[station_id] = compute_mean_by_doy(tmin_values, clamp_below_zero=False)
    return tmax_stats, tmin_stats, mean_tmax, mean_tmin


def _mixed_grid_normalize_flags(plot_specs: tuple[VariablePlotSpec, ...]) -> dict[str, bool]:
    flags = {spec.key: False for spec in plot_specs}
    flags["streamflow"] = True
    return flags


def _uniform_grid_normalize_flags(
    normalize: bool,
    plot_specs: tuple[VariablePlotSpec, ...],
) -> dict[str, bool]:
    return {spec.key: normalize for spec in plot_specs}


def _grid_temperature_ylabel(normalized: bool) -> str:
    return GRID_TEMPERATURE_YLABEL_NORMALIZED if normalized else GRID_TEMPERATURE_YLABEL


def _variable_y_label(spec: VariablePlotSpec, *, normalized: bool) -> str:
    if not normalized:
        return spec.y_label
    if spec.key == "streamflow":
        return GRID_STREAMFLOW_YLABEL_NORMALIZED
    return f"Normalized {spec.label} (z-score)"


def plot_basin_variables_grid(
    *,
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    station_ids: list[str],
    station_names: dict[str, str],
    output_path: Path,
    plot_specs: tuple[VariablePlotSpec, ...],
    only_3_months: bool,
    normalize_by_variable: dict[str, bool],
    show_grid: bool,
    show_plot: bool,
    font_size: float = FONT_SIZE,
    put_only_first_y_ticks: bool = PUT_ONLY_FIRST_Y_TICKS,
) -> None:
    n_rows = len(plot_specs)
    n_cols = len(station_ids)
    with _plot_font_rc_context(font_size):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            sharex="col",
            squeeze=False,
        )

        row_limits: list[tuple[float | None, float | None]] = []
        row_data: list[dict] = []

        for spec in plot_specs:
            var_normalize = normalize_by_variable.get(spec.key, False)
            if spec.is_temperature:
                tmax_stats, tmin_stats, mean_tmax, mean_tmin = _build_temperature_stats(
                    frames,
                    common_index,
                    normalize=var_normalize,
                )
                y_min, y_max = _temperature_y_limits(tmax_stats, tmin_stats)
                row_limits.append((y_min, y_max))
                row_data.append(
                    {
                        "type": "temperature",
                        "var_normalize": var_normalize,
                        "tmax_stats": tmax_stats,
                        "tmin_stats": tmin_stats,
                        "mean_tmax": mean_tmax,
                        "mean_tmin": mean_tmin,
                    }
                )
            else:
                station_stats, station_means = _build_single_variable_stats(
                    frames,
                    common_index,
                    spec,
                    normalize=var_normalize,
                )
                y_min, y_max = _shared_y_limits(
                    station_stats,
                    clamp_below_zero=spec.clamp_below_zero and not var_normalize,
                )
                row_limits.append((y_min, y_max))
                row_data.append(
                    {
                        "type": "single",
                        "spec": spec,
                        "var_normalize": var_normalize,
                        "station_stats": station_stats,
                        "station_means": station_means,
                    }
                )

        for row_idx, (spec, limits, data) in enumerate(zip(plot_specs, row_limits, row_data)):
            y_min, y_max = limits
            var_normalize = data["var_normalize"]
            for col_idx, station_id in enumerate(station_ids):
                axis = axes[row_idx, col_idx]
                station_label = format_station_label(station_id, station_names)
                show_title = row_idx == 0
                show_xlabel = row_idx == n_rows - 1
                show_ylabel = col_idx == 0

                if data["type"] == "temperature":
                    plot_temperature_cycle(
                        data["tmax_stats"][station_id],
                        data["tmin_stats"][station_id],
                        station_label=station_label,
                        mean_tmax=data["mean_tmax"][station_id],
                        mean_tmin=data["mean_tmin"][station_id],
                        output_path=None,
                        show_plot=False,
                        only_3_months=only_3_months,
                        clamp_below_zero=False,
                        y_min=y_min,
                        y_max=y_max,
                        normalized=var_normalize,
                        show_grid=show_grid,
                        axis=axis,
                        show_title=show_title,
                        show_xlabel=show_xlabel,
                        show_ylabel=False,
                        show_legend=False,
                    )
                    if show_title:
                        axis.set_title(station_label)
                    if show_ylabel:
                        axis.set_ylabel(_grid_temperature_ylabel(var_normalize))
                else:
                    plot_mean_cycle(
                        data["station_stats"][station_id],
                        data["spec"],
                        station_label=station_label,
                        mean_value=data["station_means"][station_id],
                        output_path=None,
                        show_plot=False,
                        only_3_months=only_3_months,
                        clamp_below_zero=spec.clamp_below_zero and not var_normalize,
                        y_min=y_min,
                        y_max=y_max,
                        normalized=var_normalize,
                        show_grid=show_grid,
                        axis=axis,
                        show_title=show_title,
                        show_xlabel=show_xlabel,
                        show_ylabel=False,
                        font_size=font_size,
                    )
                    if show_title:
                        axis.set_title(_mean_cycle_station_label(station_label), fontsize=font_size)
                    if show_ylabel:
                        axis.set_ylabel(_mean_cycle_y_label(spec, normalized=var_normalize), fontsize=font_size)
                if put_only_first_y_ticks and col_idx > 0:
                    axis.tick_params(axis="y", labelleft=False)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def format_land_use_fractions(ag: float, forest: float, shrub: float) -> str:
    if not all(np.isfinite(value) for value in (ag, forest, shrub)):
        return ""
    values = [ag, forest, shrub]
    if max(values) <= 1.0:
        values = [value * 100.0 for value in values]
    return f"{round(values[0]):.0f}/{round(values[1]):.0f}/{round(values[2]):.0f}"


def _station_summary_column_decimals(
    column: str,
    column_decimals: dict[str, int] | None,
) -> int:
    if column_decimals and column in column_decimals:
        return column_decimals[column]
    return DEFAULT_STATION_SUMMARY_DECIMALS


def _format_station_summary_number(value: float, decimals: int) -> float | str:
    if not np.isfinite(value):
        return ""
    return round(float(value), decimals)


def write_station_summary_csv(
    *,
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    static_info: pd.DataFrame,
    station_names: dict[str, str],
    output_path: Path,
    column_decimals: dict[str, int] | None = None,
) -> Path:
    rows: list[dict[str, str | float]] = []
    for station_id, frame in sorted(frames.items()):
        streamflow_col = resolve_streamflow_column(frame)
        aligned = frame.loc[common_index, streamflow_col].dropna()
        static_row = static_info.loc[station_id] if station_id in static_info.index else None
        lat = float(static_row["Latitude"]) if static_row is not None and pd.notna(static_row.get("Latitude")) else float("nan")
        lon = float(static_row["Longitude"]) if static_row is not None and pd.notna(static_row.get("Longitude")) else float("nan")
        ag = static_value(frame, STATIC_COLUMNS["agricultural"])
        forest = static_value(frame, STATIC_COLUMNS["forests"])
        shrub = static_value(frame, STATIC_COLUMNS["shrub"])
        row: dict[str, str | float] = {
            "ID": format_station_id(station_id),
            "STATION NAME": (
                _short_station_name(station_names[station_id])
                if station_id in station_names
                else station_id
            ),
            "Latitude": decimal_to_dms(lat, is_latitude=True),
            "Longitude": decimal_to_dms(lon, is_latitude=False),
            "A/F/V land use (%)": format_land_use_fractions(ag, forest, shrub),
        }
        numeric_values = {
            "Drainage Area (km2)": static_value(frame, STATIC_COLUMNS["catchment_area"]),
            "Elevation gauging station (m.a.s.l.)": static_value(
                frame,
                STATIC_COLUMNS["elevation"],
            ),
            "Mean Q (m3/s)": float(aligned.mean()) if not aligned.empty else float("nan"),
            "Q10 (m3/s)": float(np.percentile(aligned, 10)) if not aligned.empty else float("nan"),
            "Q90 (m3/s)": float(np.percentile(aligned, 90)) if not aligned.empty else float("nan"),
        }
        for column in STATION_SUMMARY_NUMERIC_COLUMNS:
            decimals = _station_summary_column_decimals(column, column_decimals)
            row[column] = _format_station_summary_number(numeric_values[column], decimals)
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, sep=";")
    return output_path


def write_gnn_lstm_variables_csv(output_path: Path) -> Path:
    rows = [
        {
            "Variable": variable,
            "GNN-LSTM I/O": io_flag,
            "Class": variable_class,
            "Resolution": resolution,
            "Source": source,
        }
        for variable, io_flag, variable_class, resolution, source in GNN_LSTM_VARIABLE_ROWS
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, sep=";")
    return output_path


def _plot_variable_set(
    *,
    frames: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    station_names: dict[str, str],
    output_root: Path,
    spec: VariablePlotSpec,
    only_3_months: bool,
    same_y_axis: bool,
    normalize: bool,
    show_grid: bool,
    show_plot: bool,
) -> int:
    variable_dir = output_root / spec.key
    variable_dir.mkdir(parents=True, exist_ok=True)
    plots_written = 0

    if spec.is_temperature:
        tmax_stats, tmin_stats, mean_tmax, mean_tmin = _build_temperature_stats(
            frames,
            common_index,
            normalize=normalize,
        )
        if same_y_axis:
            y_min, y_max = _temperature_y_limits(tmax_stats, tmin_stats)
            print(
                f"Shared {spec.key} y-axis limits across stations: "
                f"[{y_min:.4f}, {y_max:.4f}]"
            )
        else:
            y_min, y_max = None, None

        suffix = "_normalized" if normalize else ""
        for station_id in sorted(tmax_stats):
            station_label = format_station_label(station_id, station_names)
            output_path = variable_dir / f"mean_{spec.key}_{station_id}{suffix}.png"
            plot_temperature_cycle(
                tmax_stats[station_id],
                tmin_stats[station_id],
                station_label=station_label,
                mean_tmax=mean_tmax[station_id],
                mean_tmin=mean_tmin[station_id],
                output_path=output_path,
                show_plot=show_plot,
                only_3_months=only_3_months,
                clamp_below_zero=False,
                y_min=y_min,
                y_max=y_max,
                normalized=normalize,
                show_grid=show_grid,
            )
            print(f"Saved {output_path}")
            plots_written += 1
        return plots_written

    station_stats, station_means = _build_single_variable_stats(
        frames,
        common_index,
        spec,
        normalize=normalize,
    )
    clamp = spec.clamp_below_zero and not normalize
    if same_y_axis:
        y_min, y_max = _shared_y_limits(station_stats, clamp_below_zero=clamp)
        if normalize:
            print(
                f"Shared normalized {spec.key} y-axis limits across stations: "
                f"[{y_min:.4f}, {y_max:.4f}]"
            )
        else:
            print(
                f"Shared {spec.key} y-axis upper limit across stations: {y_max:.4f} {spec.unit}"
            )
    else:
        y_min, y_max = None, None

    suffix = "_normalized" if normalize else ""
    for station_id, stats in station_stats.items():
        station_label = format_station_label(station_id, station_names)
        output_path = variable_dir / f"mean_{spec.key}_{station_id}{suffix}.png"
        plot_mean_cycle(
            stats,
            spec,
            station_label=station_label,
            mean_value=station_means[station_id],
            output_path=output_path,
            show_plot=show_plot,
            only_3_months=only_3_months,
            clamp_below_zero=clamp,
            y_min=y_min if normalize else None,
            y_max=y_max,
            normalized=normalize,
            show_grid=show_grid,
        )
        print(f"Saved {output_path}")
        plots_written += 1
    return plots_written


def run_basin_summary_analysis(
    *,
    pickle_path: str | Path = DEFAULT_PICKLE_PATH,
    visuals_dir: str | Path = DEFAULT_VISUALS_DIR,
    static_info_path: str | Path = DEFAULT_STATIC_INFO_PATH,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    only_3_months: bool = ONLY_3_MONTHS,
    same_y_axis: bool = SAME_Y_AXIS,
    normalize: bool = NORMALIZE,
    show_grid: bool = SHOW_GRID,
    include_atmospheric_indexes: bool = INCLUDE_ATMOSPHERIC_INDEXES,
    font_size: float = FONT_SIZE,
    put_only_first_y_ticks: bool = PUT_ONLY_FIRST_Y_TICKS,
    show_plot: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp | int | float]]:
    frames = load_pickle_stations(pickle_path)
    common_index = common_date_index(frames)
    split_info = compute_train_test_ranges(common_index, train_fraction=train_fraction)
    print_date_summary(frames, common_index, split_info)

    station_names = load_station_name_map(static_info_path)
    static_info = load_static_info(static_info_path)
    output_root = Path(visuals_dir) / DEFAULT_OUTPUT_SUBDIR
    output_root.mkdir(parents=True, exist_ok=True)
    plot_specs = build_plot_specs(include_atmospheric_indexes=include_atmospheric_indexes)

    plots_written = 0
    plot_modes = (False, True) if normalize else (False,)
    with _plot_font_rc_context(font_size):
        for use_normalize in plot_modes:
            for spec in plot_specs:
                plots_written += _plot_variable_set(
                    frames=frames,
                    common_index=common_index,
                    station_names=station_names,
                    output_root=output_root,
                    spec=spec,
                    only_3_months=only_3_months,
                    same_y_axis=same_y_axis,
                    normalize=use_normalize,
                    show_grid=show_grid,
                    show_plot=show_plot,
                )

            grid_suffix = "_normalized" if use_normalize else ""
            grid_path = output_root / f"basin_variables_grid{grid_suffix}.png"
            plot_basin_variables_grid(
                frames=frames,
                common_index=common_index,
                station_ids=sorted(frames),
                station_names=station_names,
                output_path=grid_path,
                plot_specs=plot_specs,
                only_3_months=only_3_months,
                normalize_by_variable=_uniform_grid_normalize_flags(use_normalize, plot_specs),
                show_grid=show_grid,
                show_plot=show_plot,
                font_size=font_size,
                put_only_first_y_ticks=put_only_first_y_ticks,
            )
            print(f"Saved {grid_path}")

        mixed_grid_path = output_root / "basin_variables_grid_mixed.png"
        plot_basin_variables_grid(
            frames=frames,
            common_index=common_index,
            station_ids=sorted(frames),
            station_names=station_names,
            output_path=mixed_grid_path,
            plot_specs=plot_specs,
            only_3_months=only_3_months,
            normalize_by_variable=_mixed_grid_normalize_flags(plot_specs),
            show_grid=show_grid,
            show_plot=show_plot,
            font_size=font_size,
            put_only_first_y_ticks=put_only_first_y_ticks,
        )
        print(f"Saved {mixed_grid_path}")

    station_csv_path = write_station_summary_csv(
        frames=frames,
        common_index=common_index,
        static_info=static_info,
        station_names=station_names,
        output_path=output_root / "station_summary.csv",
    )
    variables_csv_path = write_gnn_lstm_variables_csv(output_root / "gnn_lstm_variables.csv")
    print(f"Saved {station_csv_path}")
    print(f"Saved {variables_csv_path}")
    print(f"\nWrote {plots_written} variable plots to {output_root}")
    return summarize_station_date_ranges(frames), split_info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize pickle date ranges, report an 80/20 train-test day split, "
            "and generate basin summary graphics for streamflow and related variables."
        )
    )
    parser.add_argument(
        "--pickle-path",
        default=DEFAULT_PICKLE_PATH,
        help=f"Input pickle path (default: {DEFAULT_PICKLE_PATH}).",
    )
    parser.add_argument(
        "--visuals-dir",
        default=DEFAULT_VISUALS_DIR,
        help=(
            f"Visuals root directory; output goes to "
            f"{{visuals_dir}}/{DEFAULT_OUTPUT_SUBDIR} (default: {DEFAULT_VISUALS_DIR})."
        ),
    )
    parser.add_argument(
        "--static-info-path",
        default=DEFAULT_STATIC_INFO_PATH,
        help=f"Static info CSV for station names (default: {DEFAULT_STATIC_INFO_PATH}).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help=f"Train fraction for day-range summary (default: {DEFAULT_TRAIN_FRACTION}).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots interactively.",
    )
    parser.add_argument(
        "--only-3-months",
        action="store_true",
        default=ONLY_3_MONTHS,
        help="Show only start, middle, and end month labels on the x-axis.",
    )
    parser.add_argument(
        "--no-same-y-axis",
        action="store_false",
        dest="same_y_axis",
        help="Use independent y-axis limits for each station plot.",
    )
    parser.set_defaults(same_y_axis=SAME_Y_AXIS)
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=NORMALIZE,
        help=(
            "Also write z-score normalized plots and a normalized grid with a _normalized suffix."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Write only raw (physical-unit) plots and grid.",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        default=SHOW_GRID,
        help="Show grid lines on the plots.",
    )
    parser.add_argument(
        "--include-atmospheric-indexes",
        action="store_true",
        default=INCLUDE_ATMOSPHERIC_INDEXES,
        help="Include SPEI, NAO, and WEMO in plots and grid (default: enabled).",
    )
    parser.add_argument(
        "--no-atmospheric-indexes",
        action="store_false",
        dest="include_atmospheric_indexes",
        help="Exclude atmospheric index plots from output.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=FONT_SIZE,
        help=f"Font size for all plot text (default: {FONT_SIZE}).",
    )
    parser.add_argument(
        "--put-only-first-y-ticks",
        action="store_true",
        default=PUT_ONLY_FIRST_Y_TICKS,
        help="Show y-axis tick labels only on the first column of grid plots.",
    )
    parser.add_argument(
        "--all-y-ticks",
        action="store_false",
        dest="put_only_first_y_ticks",
        help="Show y-axis tick labels on every column of grid plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_basin_summary_analysis(
        pickle_path=args.pickle_path,
        visuals_dir=args.visuals_dir,
        static_info_path=args.static_info_path,
        train_fraction=args.train_fraction,
        only_3_months=args.only_3_months,
        same_y_axis=args.same_y_axis,
        normalize=args.normalize,
        show_grid=args.show_grid,
        include_atmospheric_indexes=args.include_atmospheric_indexes,
        font_size=args.font_size,
        put_only_first_y_ticks=args.put_only_first_y_ticks,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
