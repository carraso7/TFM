#!/usr/bin/env python3
from __future__ import annotations

"""
Code for creating pickle and/or semicolon-separated text station files,
filtering by station ids or graph edges.
It also writes the necessary files for the return period analysis.
"""

import argparse
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from create_static_info_data_treatment import (
    DEFAULT_FALLBACK_STATIONS,
    _normalize_station_id,
    _parse_station_id_from_filename,
)

# Example execution: python code/data_processing.py --end-date 2020-12-31

DEFAULT_OUTPUT_PATH = r"/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
DEFAULT_RETURN_PERIOD_TXT_NAME = "inputs_selected_stations_for_ret_period.txt"
DEFAULT_STATIC_INFO_PATH = r"/mnt/d/streamflow_prediction/static_info.csv"
DEFAULT_MERGE_LOG_NAME = "input_merge_dynamic_differences.csv"
DEFAULT_MERGE_STATIC_LOG_NAME = "input_merge_static_differences.csv"
DEFAULT_START_DATE = "1991-10-01"
# When the same station/date appears in multiple input paths, values from the
# last matching path in this list take precedence.
DEFAULT_INPUT_PATHS = [
    r"/mnt/d/streamflow_prediction/inputs_allstations_plus_static.pkl",
    # r"/mnt/d/streamflow_prediction/inputs_allstations_plus_static_checked.pkl",
    r"/mnt/d/streamflow_prediction/inputs_allstations_plus_static_checked_v2.pkl",
    # r"/mnt/d/streamflow_prediction/inputs_data_A271.txt",
    r"/mnt/d/streamflow_prediction/inputs_data_A271_artificially_completed.txt",
]
DEFAULT_EDGES: list[tuple[str, str]] = [
    ("061", "170"),
    ("018", "170"),
    ("080", "062"),
    ("062", "170"),
    ("271", "018"),
]


def _edges_to_tokens(edges: list[tuple[str, str]]) -> list[str]:
    """Convert ``(source, target)`` edge pairs to CLI-friendly comma tokens."""
    return [f"{source},{target}" for source, target in edges]

SHORT_VALUE_COLUMNS = [
    "Streamflow",
    "pr",
    "tmax_total",
    "tmin_total",
    "Humidity",
    "SPEI",
    "nao",
    "WEMO",
]
LONG_VALUE_COLUMNS = [
    "Streamflow [m3/s]",
    "Catchment-averaged precipitation [mm]",
    "Maximum temperature [°C]",
    "Minimum temperature [°C]",
    "Air humidity [%]",
    "SPEI [−]",
    "NAO [−]",
    "WeMO [−]",
]
STATIC_COLUMN_NAMES = {
    "Catchment Area (km2)",
    "Elevation gauging station (m.a.s.l.)",
    "Agricultural areas",
    "Forests",
    "Shrub and/or herbaceous vegetation",
    "Catchment area",
    "Elevation",
    "Agricultural area (%)",
    "Forestal area (%)",
    "Shrub area (%)",
}
TEXT_DATA_COLUMNS = [
    "Streamflow",
    "pr",
    "tmax_total",
    "tmin_total",
    "Humidity",
    "SPEI",
    "nao",
    "WEMO",
    "Catchment Area (km2)",
    "Elevation gauging station (m.a.s.l.)",
    "Agricultural areas",
    "Forests",
    "Shrub and/or herbaceous vegetation",
]
STATIC_INFO_COLUMNS = [
    "Catchment Area (km2)",
    "Elevation gauging station (m.a.s.l.)",
    "Agricultural areas",
    "Forests",
    "Shrub and/or herbaceous vegetation",
]
MERGE_COMPARE_ATOL = 1e-9
MERGE_COMPARE_RTOL = 1e-9


def _parse_station_id_from_input_path(path: Path) -> str | None:
    """Infer a station id from an input filename stem.

    Tries the shared filename parser first, then ``A###`` and generic
    three-digit patterns.

    Args:
        path: Input pickle or text file path.

    Returns:
        Normalized three-digit station id, or ``None`` if not found.
    """
    station_id = _parse_station_id_from_filename(path)
    if station_id:
        return station_id

    stem = path.stem
    match = re.search(r"A(\d{3})", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(\d{3})", stem)
    if match:
        return match.group(1)
    return None


def _iter_station_frames(data: dict[str, Any]) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield DataFrame entries from a pickle station dictionary."""
    for station_id, value in data.items():
        if isinstance(value, pd.DataFrame):
            yield str(station_id), value


def _find_text_header_line(path: Path) -> int:
    """Locate the zero-based header line in a SAIH semicolon-separated text file.

    Args:
        path: Station export ``.txt`` or ``.csv`` file.

    Returns:
        Line number of the data header row.

    Raises:
        ValueError: If no recognizable header line is found.
    """
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("serie de tiempo") and ";" in lower:
                return line_number
            if stripped.startswith(";") and "streamflow" in lower:
                return line_number
    raise ValueError(f"Could not find a data header in text file: {path}")


def _load_text_station_file(path: Path, station_id: str | None = None) -> tuple[str, pd.DataFrame]:
    """Load one semicolon-separated SAIH station file into a DataFrame.

    Args:
        path: Text file with header metadata followed by a daily data table.
        station_id: Optional explicit station id override.

    Returns:
        Tuple ``(station_id, dataframe)`` indexed by date with columns from
        ``TEXT_DATA_COLUMNS``.

    Raises:
        ValueError: If the station id or required columns cannot be resolved.
    """
    resolved_station_id = station_id or _parse_station_id_from_input_path(path)
    if not resolved_station_id:
        raise ValueError(
            f"Could not infer station id from filename {path.name}. "
            "Use a filename containing the station code or pass --station-id."
        )
    resolved_station_id = _normalize_station_id(resolved_station_id)

    header_line = _find_text_header_line(path)
    df = pd.read_csv(
        path,
        sep=";",
        skiprows=header_line,
        encoding="latin-1",
        decimal=".",
        engine="python",
    )
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df.index.name = None

    missing_columns = [col for col in TEXT_DATA_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Text file {path} is missing columns: {missing_columns}")

    return resolved_station_id, df[TEXT_DATA_COLUMNS]


def _load_pickle_file(path: Path) -> dict[str, pd.DataFrame]:
    """Load a pickle containing ``dict[str, pd.DataFrame]`` station data.

    Args:
        path: Pickle file path.

    Returns:
        Dict keyed by string station id with DataFrame values only.

    Raises:
        TypeError: If the pickle root object is not a dict of DataFrames.
    """
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Pickle file {path} must contain a dict[str, DataFrame], got {type(data)}")
    return {str(station_id): df for station_id, df in _iter_station_frames(data)}


def _align_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the index to midnight datetimes and sort chronologically."""
    aligned_df = df.copy()
    if not isinstance(aligned_df.index, pd.DatetimeIndex):
        aligned_df.index = pd.to_datetime(aligned_df.index, errors="coerce")
    aligned_df = aligned_df.loc[aligned_df.index.notna()]
    aligned_df.index = aligned_df.index.normalize()
    return aligned_df.sort_index()


def _dynamic_columns_in_df(df: pd.DataFrame) -> list[str]:
    """Return dynamic value columns present in a station DataFrame."""
    for candidates in (SHORT_VALUE_COLUMNS, LONG_VALUE_COLUMNS):
        present = [column for column in candidates if column in df.columns]
        if present:
            return present
    return [
        column
        for column in df.columns
        if column not in STATIC_COLUMN_NAMES
    ]


def _static_value_from_frame(df: pd.DataFrame, column: str) -> float | None:
    """Read a single time-invariant scalar from the first valid row of a column."""
    if column not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna().unique()
    if values.size == 0:
        return None
    return float(values[0])


@dataclass
class StaticMergeDifference:
    """Record of a static attribute mismatch when merging input files."""

    station_id: str
    column: str
    earlier_files: str
    later_file: str
    stored_file: str
    earlier_value: float | None
    later_value: float | None


@dataclass
class InputMergeLog:
    """Accumulator for dynamic and static differences found while merging inputs."""

    dynamic_rows: list[dict[str, Any]] = field(default_factory=list)
    static_differences: list[StaticMergeDifference] = field(default_factory=list)

    def record_merge(
        self,
        *,
        station_id: str,
        earlier_files: list[str],
        later_file: str,
        existing: pd.DataFrame,
        new_frame: pd.DataFrame,
        atol: float = MERGE_COMPARE_ATOL,
        rtol: float = MERGE_COMPARE_RTOL,
    ) -> None:
        """Compare overlapping station frames and record dynamic/static differences."""
        earlier_label = "; ".join(earlier_files)
        stored_file = later_file
        existing_aligned = _align_dataframe(existing)
        new_aligned = _align_dataframe(new_frame)
        overlap = existing_aligned.index.intersection(new_aligned.index)
        earlier_only_dates = existing_aligned.index.difference(new_aligned.index)
        later_only_dates = new_aligned.index.difference(existing_aligned.index)

        for column in _dynamic_columns_in_df(existing_aligned):
            if column not in new_aligned.columns:
                self.dynamic_rows.append(
                    {
                        "station_id": station_id,
                        "earlier_files": earlier_label,
                        "later_file": later_file,
                        "stored_file": stored_file,
                        "column": column,
                        "overlapping_dates": len(overlap),
                        "earlier_only_dates": len(earlier_only_dates),
                        "later_only_dates": len(later_only_dates),
                        "differing_days": np.nan,
                        "max_abs_difference": np.nan,
                        "mean_abs_difference": np.nan,
                        "rmse_on_common_dates": np.nan,
                        "missing_in_later_file": True,
                    }
                )
                continue

            if overlap.empty:
                differing_days = 0
                max_abs_difference = np.nan
                mean_abs_difference = np.nan
                rmse_on_common_dates = np.nan
            else:
                earlier_series = pd.to_numeric(
                    existing_aligned.loc[overlap, column],
                    errors="coerce",
                )
                later_series = pd.to_numeric(
                    new_aligned.loc[overlap, column],
                    errors="coerce",
                )
                valid_mask = earlier_series.notna() & later_series.notna()
                earlier_valid = earlier_series[valid_mask]
                later_valid = later_series[valid_mask]

                if earlier_valid.empty:
                    differing_days = 0
                    max_abs_difference = np.nan
                    mean_abs_difference = np.nan
                    rmse_on_common_dates = np.nan
                else:
                    diff = later_valid - earlier_valid
                    differs = ~np.isclose(
                        earlier_valid.to_numpy(),
                        later_valid.to_numpy(),
                        atol=atol,
                        rtol=rtol,
                    )
                    differing_days = int(differs.sum())
                    max_abs_difference = float(np.abs(diff).max())
                    mean_abs_difference = float(np.abs(diff).mean())
                    rmse_on_common_dates = float(np.sqrt(np.mean(diff**2)))

            self.dynamic_rows.append(
                {
                    "station_id": station_id,
                    "earlier_files": earlier_label,
                    "later_file": later_file,
                    "stored_file": stored_file,
                    "column": column,
                    "overlapping_dates": len(overlap),
                    "earlier_only_dates": len(earlier_only_dates),
                    "later_only_dates": len(later_only_dates),
                    "differing_days": differing_days,
                    "max_abs_difference": max_abs_difference,
                    "mean_abs_difference": mean_abs_difference,
                    "rmse_on_common_dates": rmse_on_common_dates,
                    "missing_in_later_file": False,
                }
            )

        for column in STATIC_INFO_COLUMNS:
            earlier_value = _static_value_from_frame(existing_aligned, column)
            later_value = _static_value_from_frame(new_aligned, column)
            values_differ = (
                earlier_value is not None
                and later_value is not None
                and not np.isclose(earlier_value, later_value, atol=atol, rtol=rtol)
            )
            if values_differ:
                self.static_differences.append(
                    StaticMergeDifference(
                        station_id=station_id,
                        column=column,
                        earlier_files=earlier_label,
                        later_file=later_file,
                        stored_file=stored_file,
                        earlier_value=earlier_value,
                        later_value=later_value,
                    )
                )

    def has_dynamic_differences(self) -> bool:
        """Return ``True`` if any dynamic column has differing overlapping days."""
        return any(
            isinstance(row.get("differing_days"), (int, np.integer))
            and int(row["differing_days"]) > 0
            for row in self.dynamic_rows
        )

    def has_static_differences(self) -> bool:
        """Return ``True`` if any static attribute mismatch was recorded."""
        return bool(self.static_differences)

    def write_dynamic_csv(self, output_path: Path) -> Path | None:
        """Write dynamic merge comparisons to CSV if any rows were recorded.

        Returns:
            Output path when written, otherwise ``None``.
        """
        if not self.dynamic_rows:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.dynamic_rows).to_csv(output_path, index=False)
        return output_path

    def write_static_csv(self, output_path: Path) -> Path | None:
        """Write static merge differences to CSV if any were recorded.

        Returns:
            Output path when written, otherwise ``None``.
        """
        if not self.static_differences:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "station_id": diff.station_id,
                "column": diff.column,
                "earlier_files": diff.earlier_files,
                "later_file": diff.later_file,
                "stored_file": diff.stored_file,
                "earlier_value": diff.earlier_value,
                "later_value": diff.later_value,
                "difference": diff.later_value - diff.earlier_value,
                "abs_difference": abs(diff.later_value - diff.earlier_value),
                "values_differ": True,
            }
            for diff in self.static_differences
        ]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return output_path

    def print_static_differences(self) -> None:
        """Print a human-readable summary of static merge conflicts."""
        if not self.static_differences:
            print("\nStatic info: no differences found while merging input files.")
            return

        print("\nStatic info differences while merging input files:")
        for diff in self.static_differences:
            print(f"- Station {diff.station_id} | {diff.column}")
            print(
                f"    earlier ({diff.earlier_files}): "
                f"{diff.earlier_value!r}"
            )
            print(
                f"    later   ({diff.later_file}): "
                f"{diff.later_value!r}"
            )
            print(f"    kept from: {diff.stored_file}")


def _merge_station_frames(
    existing: pd.DataFrame,
    new_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Merge two station frames; overlapping dates use values from new_frame."""
    existing_aligned = _align_dataframe(existing)
    new_aligned = _align_dataframe(new_frame)
    combined = pd.concat([existing_aligned, new_aligned])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def load_input_files(
    paths: list[Path],
    station_id_override: str | None = None,
) -> tuple[dict[str, pd.DataFrame], InputMergeLog]:
    """
    Load and merge station data from multiple input files.

    Files are processed in list order. When a station appears in more than one
    input, dates that overlap are updated with values from the last file in
    the list; non-overlapping dates from earlier files are preserved.
    """
    merged: dict[str, pd.DataFrame] = {}
    station_source_files: dict[str, list[str]] = {}
    merge_log = InputMergeLog()

    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".pkl", ".pickle", ".pckl"}:
            frames = _load_pickle_file(path)
        elif suffix in {".txt", ".csv"}:
            station_id, frame = _load_text_station_file(path, station_id=station_id_override)
            frames = {station_id: frame}
        else:
            raise ValueError(f"Unsupported input file type: {path}")

        for station_id, frame in frames.items():
            normalized_id = _normalize_station_id(station_id)
            if normalized_id in merged:
                merge_log.record_merge(
                    station_id=normalized_id,
                    earlier_files=station_source_files[normalized_id],
                    later_file=path.name,
                    existing=merged[normalized_id],
                    new_frame=frame,
                )
                merged[normalized_id] = _merge_station_frames(
                    merged[normalized_id],
                    frame,
                )
                if path.name not in station_source_files[normalized_id]:
                    station_source_files[normalized_id].append(path.name)
            else:
                merged[normalized_id] = frame.copy()
                station_source_files[normalized_id] = [path.name]

    if not merged:
        raise ValueError("No station data loaded from the provided input files")
    return merged, merge_log


def _parse_edge_token(token: str) -> tuple[str, str]:
    """Parse a graph edge token in ``source,target`` or ``source->target`` form.

    Raises:
        ValueError: If the token cannot be parsed into two station ids.
    """
    token = token.strip()
    if "->" in token:
        source, target = token.split("->", 1)
    elif "," in token:
        source, target = token.split(",", 1)
    else:
        raise ValueError(f"Could not parse edge token: {token!r}")
    source = _normalize_station_id(source.strip())
    target = _normalize_station_id(target.strip())
    if not source or not target:
        raise ValueError(f"Invalid edge token: {token!r}")
    return source, target


def _looks_like_edge(token: str) -> bool:
    """Return ``True`` if a CLI token appears to describe a directed edge."""
    token = token.strip()
    return "->" in token or "," in token


def parse_stations_or_edges(tokens: list[str]) -> tuple[list[str] | None, list[tuple[str, str]] | None]:
    """Parse CLI tokens as either station ids or graph edges.

    Args:
        tokens: Station ids and/or edge tokens from the command line.

    Returns:
        Tuple ``(station_ids, edges)`` where exactly one entry is ``None``.

    Raises:
        ValueError: If no tokens are supplied or an edge token is invalid.
    """
    if not tokens:
        raise ValueError("At least one station id or edge must be provided")

    if any(_looks_like_edge(token) for token in tokens):
        edges = [_parse_edge_token(token) for token in tokens]
        return None, edges

    station_ids = [_normalize_station_id(token) for token in tokens]
    return station_ids, None


def resolve_station_ids(
    station_ids: list[str] | None,
    edges: list[tuple[str, str]] | None,
) -> list[str]:
    """Resolve the final sorted station list from ids or edge endpoints."""
    if station_ids is not None:
        return sorted(dict.fromkeys(station_ids))

    if edges is None:
        raise ValueError("Either station ids or edges must be provided")

    nodes: list[str] = []
    seen: set[str] = set()
    for source, target in edges:
        for node in (source, target):
            if node not in seen:
                seen.add(node)
                nodes.append(node)
    return sorted(nodes)


def _resolve_available_station_id(requested_id: str, available_ids: set[str]) -> str | None:
    """Match a requested station id to an available key, ignoring leading zeros."""
    normalized = _normalize_station_id(requested_id)
    if normalized in available_ids:
        return normalized

    stripped = normalized.lstrip("0") or "0"
    for candidate in available_ids:
        if candidate.lstrip("0") or "0" == stripped:
            return candidate
    return None


def filter_station_data(
    data: dict[str, pd.DataFrame],
    requested_station_ids: list[str],
) -> dict[str, pd.DataFrame]:
    """Subset a station dictionary to the requested ids.

    Raises:
        ValueError: If any requested station is missing from ``data``.
    """
    available_ids = set(data.keys())
    filtered: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for requested_id in requested_station_ids:
        resolved_id = _resolve_available_station_id(requested_id, available_ids)
        if resolved_id is None:
            missing.append(requested_id)
            continue
        filtered[resolved_id] = data[resolved_id].copy()

    if missing:
        raise ValueError(
            "Requested stations not found in input data: "
            + ", ".join(missing)
            + f". Available stations: {', '.join(sorted(available_ids))}"
        )
    return filtered


MAX_MISSING_VALUE_DATES_TO_PRINT = 25


@dataclass(frozen=True)
class MissingValueDetail:
    """One missing dynamic value at a specific station date."""

    column: str
    date: pd.Timestamp


@dataclass(frozen=True)
class StationSummary:
    """Coverage and missing-data summary for one station."""

    station_id: str
    name: str
    first_day: pd.Timestamp
    last_day: pd.Timestamp
    missing_days: int
    missing_dates: tuple[pd.Timestamp, ...]
    missing_value_details: tuple[MissingValueDetail, ...]


def _value_columns(df: pd.DataFrame) -> list[str]:
    """Return dynamic value columns present in a station DataFrame."""
    for candidates in (SHORT_VALUE_COLUMNS, LONG_VALUE_COLUMNS):
        present = [column for column in candidates if column in df.columns]
        if present:
            return present
    return [
        column
        for column in df.columns
        if column not in STATIC_COLUMN_NAMES
    ]


def load_station_name_map(static_info_path: str | Path) -> dict[str, str]:
    """Load ``station_id -> Station name`` mappings from ``static_info.csv``.

    Args:
        static_info_path: CSV exported by the static-info builder.

    Returns:
        Dict keyed by normalized station id. Returns an empty dict if the file
        or required columns are missing.
    """
    static_info_path = Path(static_info_path)
    if not static_info_path.exists():
        return {}

    static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
    if "station_id" not in static_info_df.columns:
        static_info_df = static_info_df.reset_index()
    if "Station name" not in static_info_df.columns:
        return {}

    station_names: dict[str, str] = {}
    for _, row in static_info_df.iterrows():
        station_id = row.get("station_id")
        name = row.get("Station name")
        if pd.isna(station_id) or pd.isna(name):
            continue
        station_names[_normalize_station_id(str(station_id))] = str(name).strip()
    return station_names


def _format_day(day: pd.Timestamp) -> str:
    """Format a timestamp as ``YYYY-MM-DD``."""
    return day.strftime("%Y-%m-%d")


def _parse_optional_date(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Parse an optional date string or timestamp to a normalized midnight value.

    Raises:
        ValueError: If a non-null value cannot be parsed as a date.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value!r}. Use YYYY-MM-DD format.")
    return pd.Timestamp(parsed).normalize()


def filter_by_date_bounds(
    data: dict[str, pd.DataFrame],
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    """Restrict each station frame to an optional inclusive date range.

    Raises:
        ValueError: If ``start_date`` is after ``end_date``.
    """
    if start_date is None and end_date is None:
        return {station_id: df.copy() for station_id, df in data.items()}
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError(
            f"start-date ({_format_day(start_date)}) must be on or before "
            f"end-date ({_format_day(end_date)})"
        )

    filtered: dict[str, pd.DataFrame] = {}
    for station_id, df in data.items():
        aligned_df = _align_dataframe(df)
        if start_date is not None:
            aligned_df = aligned_df.loc[aligned_df.index >= start_date]
        if end_date is not None:
            aligned_df = aligned_df.loc[aligned_df.index <= end_date]
        filtered[station_id] = aligned_df
    return filtered


def compute_common_interval(
    data: dict[str, pd.DataFrame],
    station_ids: list[str],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Compute the largest date interval common to all listed stations.

    Returns:
        ``(start, end)`` inclusive common interval, or ``None`` if stations
        do not overlap.
    """
    first_days: list[pd.Timestamp] = []
    last_days: list[pd.Timestamp] = []

    for station_id in station_ids:
        df = data.get(station_id)
        if df is None or df.empty:
            continue
        aligned_df = _align_dataframe(df)
        if aligned_df.empty:
            continue
        first_days.append(aligned_df.index.min())
        last_days.append(aligned_df.index.max())

    if not first_days:
        return None

    common_start = max(first_days)
    common_end = min(last_days)
    if common_start > common_end:
        return None
    return common_start, common_end


def _resolve_station_name(station_id: str, station_names: dict[str, str]) -> str:
    """Resolve a display name from static info or fallback station metadata."""
    normalized_id = _normalize_station_id(station_id)
    if normalized_id in station_names:
        return station_names[normalized_id]

    fallback = DEFAULT_FALLBACK_STATIONS.get(normalized_id)
    if fallback is not None:
        return fallback.name

    return "Unknown"


def summarize_station_dataframe(df: pd.DataFrame) -> tuple[
    pd.Timestamp | None,
    pd.Timestamp | None,
    int,
    tuple[pd.Timestamp, ...],
    tuple[MissingValueDetail, ...],
]:
    """Summarize date coverage, missing calendar days, and missing value cells."""
    if df.empty:
        return None, None, 0, (), ()

    aligned_df = _align_dataframe(df)
    if aligned_df.empty:
        return None, None, 0, (), ()

    first_day = aligned_df.index.min()
    last_day = aligned_df.index.max()
    full_range = pd.date_range(first_day, last_day, freq="D")
    present_dates = pd.DatetimeIndex(aligned_df.index).unique()
    missing_dates_index = full_range.difference(present_dates)
    missing_dates = tuple(pd.Timestamp(date).normalize() for date in missing_dates_index)
    missing_days = len(missing_dates)

    value_columns = _value_columns(aligned_df)
    df_in_span = aligned_df.loc[
        (aligned_df.index >= first_day) & (aligned_df.index <= last_day)
    ]
    missing_value_details: list[MissingValueDetail] = []
    for column in value_columns:
        nan_dates = df_in_span.index[df_in_span[column].isna()]
        for date in nan_dates:
            missing_value_details.append(
                MissingValueDetail(column=column, date=pd.Timestamp(date).normalize())
            )

    return first_day, last_day, missing_days, missing_dates, tuple(missing_value_details)


def build_station_summaries(
    data: dict[str, pd.DataFrame],
    station_ids: list[str],
    static_info_path: str | Path = DEFAULT_STATIC_INFO_PATH,
) -> list[StationSummary]:
    """Build printable coverage summaries for the requested stations."""
    station_names = load_station_name_map(static_info_path)
    summaries: list[StationSummary] = []

    for station_id in station_ids:
        df = data.get(station_id)
        if df is None:
            continue
        first_day, last_day, missing_days, missing_dates, missing_value_details = (
            summarize_station_dataframe(df)
        )
        if first_day is None or last_day is None:
            continue
        summaries.append(
            StationSummary(
                station_id=station_id,
                name=_resolve_station_name(station_id, station_names),
                first_day=first_day,
                last_day=last_day,
                missing_days=missing_days,
                missing_dates=missing_dates,
                missing_value_details=missing_value_details,
            )
        )
    return summaries


def _format_date_list(dates: tuple[pd.Timestamp, ...], limit: int = MAX_MISSING_VALUE_DATES_TO_PRINT) -> str:
    """Format a tuple of dates for console output with optional truncation."""
    if not dates:
        return "none"
    formatted = [_format_day(date) for date in dates[:limit]]
    if len(dates) > limit:
        formatted.append(f"... and {len(dates) - limit} more")
    return ", ".join(formatted)


def _format_missing_value_details(details: tuple[MissingValueDetail, ...]) -> list[str]:
    """Format missing dynamic value cells grouped by column."""
    if not details:
        return ["  missing value cells: none"]

    by_column: dict[str, list[pd.Timestamp]] = {}
    for detail in details:
        by_column.setdefault(detail.column, []).append(detail.date)

    lines = ["  missing value cells:"]
    for column in sorted(by_column):
        dates = tuple(sorted(by_column[column]))
        lines.append(f"    - {column}: {_format_date_list(dates)}")
    return lines


def print_station_summaries(
    summaries: list[StationSummary],
    common_interval: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> None:
    """Print station summaries and the common overlapping interval."""
    print("\nNode summary:")
    for summary in summaries:
        print(
            f"- {summary.station_id} | {summary.name} | "
            f"{_format_day(summary.first_day)} to {_format_day(summary.last_day)} | "
            f"missing days: {summary.missing_days}"
        )
        if summary.missing_days:
            print(f"  missing dates: {_format_date_list(summary.missing_dates)}")
        for line in _format_missing_value_details(summary.missing_value_details):
            print(line)

    print("\nBiggest common interval across all stations:")
    if common_interval is None:
        print("- none (stations do not overlap)")
    else:
        common_start, common_end = common_interval
        common_days = (common_end - common_start).days + 1
        print(
            f"- {_format_day(common_start)} to {_format_day(common_end)} "
            f"({common_days} days)"
        )


def write_output_pickle(data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Write ``dict[str, pd.DataFrame]`` station data to a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _streamflow_column(df: pd.DataFrame) -> str:
    """Return the streamflow column name supported by a station DataFrame.

    Raises:
        ValueError: If neither short nor long streamflow column names exist.
    """
    if "Streamflow" in df.columns:
        return "Streamflow"
    if "Streamflow [m3/s]" in df.columns:
        return "Streamflow [m3/s]"
    raise ValueError(
        "DataFrame must include a streamflow column: 'Streamflow' or 'Streamflow [m3/s]'"
    )


def _format_return_period_station_name(name: str) -> str:
    """Normalize a station name for return-period text export."""
    formatted = name.strip()
    if formatted.lower().startswith("rio "):
        formatted = formatted[4:]
    return formatted.upper()


def _resolve_return_period_station_metadata(
    station_ids: list[str],
    station_names: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Build parallel catchment and display-name lists for return-period export."""
    catchments: list[str] = []
    names: list[str] = []

    for station_id in station_ids:
        normalized_id = _normalize_station_id(station_id)
        name = _format_return_period_station_name(
            _resolve_station_name(normalized_id, station_names)
        )
        catchment = name.split(" en ", 1)[0] if " en " in name else ""
        catchments.append(catchment)
        names.append(name)

    return catchments, names


def _build_return_period_streamflow_table(
    data: dict[str, pd.DataFrame],
    station_ids: list[str],
) -> pd.DataFrame:
    """Assemble a wide daily streamflow table for return-period analysis."""
    all_dates: pd.DatetimeIndex | None = None
    for station_id in station_ids:
        aligned_df = _align_dataframe(data[station_id])
        if aligned_df.empty:
            continue
        all_dates = (
            aligned_df.index
            if all_dates is None
            else all_dates.union(aligned_df.index)
        )

    if all_dates is None or all_dates.empty:
        raise ValueError("No dated streamflow records available for return-period export")

    streamflow = pd.DataFrame(index=all_dates.sort_values())
    for station_id in station_ids:
        aligned_df = _align_dataframe(data[station_id])
        streamflow_column = _streamflow_column(aligned_df)
        streamflow[station_id] = aligned_df[streamflow_column]
    return streamflow


def return_period_txt_path_from_pickle(output_pickle_path: str | Path) -> Path:
    """Derive the default return-period text path from an output pickle location."""
    return Path(output_pickle_path).parent / DEFAULT_RETURN_PERIOD_TXT_NAME


def write_return_period_txt(
    data: dict[str, pd.DataFrame],
    station_ids: list[str],
    output_path: str | Path,
    static_info_path: str | Path = DEFAULT_STATIC_INFO_PATH,
) -> Path:
    """Write the semicolon-separated return-period streamflow text file.

    Output file structure:
        Row 1: ``Time;<station_ids...>``
        Row 2: ``station catchment;<catchment names...>``
        Row 3: ``station name;<station names...>``
        Remaining rows: ``YYYY-MM-DD;<streamflow values...>``

    Args:
        data: Filtered station dictionary with daily streamflow columns.
        station_ids: Ordered station columns to export.
        output_path: Destination ``.txt`` path.
        static_info_path: CSV used to resolve station names and catchments.

    Returns:
        Path to the written text file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    station_names = load_station_name_map(static_info_path)
    catchments, names = _resolve_return_period_station_metadata(
        station_ids,
        station_names,
    )
    streamflow = _build_return_period_streamflow_table(data, station_ids)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Time;" + ";".join(station_ids) + "\n")
        handle.write("station catchment;" + ";".join(catchments) + "\n")
        handle.write("station name;" + ";".join(names) + "\n")
        for date, row in streamflow.iterrows():
            values: list[str] = []
            for station_id in station_ids:
                value = row[station_id]
                if pd.isna(value):
                    values.append("")
                else:
                    values.append(str(float(value)))
            handle.write(f"{_format_day(pd.Timestamp(date))};" + ";".join(values) + "\n")

    return output_path


def _report_input_merge_differences(
    merge_log: InputMergeLog,
    dynamic_log_path: Path,
    static_log_path: Path,
    num_input_files: int,
) -> None:
    """Print and optionally write merge-difference logs for multi-file inputs."""
    merge_log.print_static_differences()

    if num_input_files <= 1:
        return

    if merge_log.has_static_differences():
        written_static_path = merge_log.write_static_csv(static_log_path)
        if written_static_path is not None:
            print(
                f"\nStatic info: wrote {len(merge_log.static_differences)} "
                f"static difference(s) to {written_static_path}"
            )

    if not merge_log.dynamic_rows:
        return

    if not merge_log.has_dynamic_differences():
        print(
            "\nDynamic info: overlapping station dates were found between input files, "
            "but no dynamic value differences were detected."
        )
        return

    written_dynamic_path = merge_log.write_dynamic_csv(dynamic_log_path)
    if written_dynamic_path is None:
        return

    total_differing_days = sum(
        int(row["differing_days"])
        for row in merge_log.dynamic_rows
        if isinstance(row.get("differing_days"), (int, np.integer))
    )
    affected_rows = sum(
        1
        for row in merge_log.dynamic_rows
        if isinstance(row.get("differing_days"), (int, np.integer))
        and int(row["differing_days"]) > 0
    )
    print("\nDynamic info differences while merging input files:")
    print(f"- station/column comparisons with differences: {affected_rows}")
    print(f"- total differing days across comparisons: {total_differing_days}")
    print(f"- stored values on overlaps come from the later file listed in each CSV row")
    print(f"- wrote dynamic merge log to {written_dynamic_path}")


def process_data(
    input_paths: list[str | Path],
    stations_or_edges: list[str],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    station_id_override: str | None = None,
    static_info_path: str | Path = DEFAULT_STATIC_INFO_PATH,
    start_date: str | pd.Timestamp | None = DEFAULT_START_DATE,
    end_date: str | pd.Timestamp | None = None,
    return_period_txt_path: str | Path | None = None,
    merge_log_path: str | Path | None = None,
    merge_static_log_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load, merge, filter, and export selected station data.

    Writes both the filtered pickle and the companion return-period text file,
    then prints station coverage summaries.

    Args:
        input_paths: One or more pickle or text station files to merge.
        stations_or_edges: Station ids or edge tokens defining the output subset.
        output_path: Destination pickle path.
        station_id_override: Optional id override for a single text input.
        static_info_path: CSV with station names for summaries and export.
        start_date: Optional first stored day (inclusive).
        end_date: Optional last stored day (inclusive).
        return_period_txt_path: Optional override for the return-period txt path.
        merge_log_path: Optional CSV path for dynamic merge differences.
        merge_static_log_path: Optional CSV path for static merge differences.

    Returns:
        Filtered ``dict[str, pd.DataFrame]`` written to ``output_path``.
    """
    paths = [Path(path) for path in input_paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    merged_data, merge_log = load_input_files(paths, station_id_override=station_id_override)
    station_ids, edges = parse_stations_or_edges(stations_or_edges)
    requested_station_ids = resolve_station_ids(station_ids, edges)
    filtered_data = filter_station_data(merged_data, requested_station_ids)

    parsed_start_date = _parse_optional_date(start_date)
    parsed_end_date = _parse_optional_date(end_date)

    output_data = filter_by_date_bounds(
        filtered_data,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
    )

    output_path = Path(output_path)
    resolved_dynamic_log_path = (
        Path(merge_log_path)
        if merge_log_path is not None
        else output_path.parent / DEFAULT_MERGE_LOG_NAME
    )
    resolved_static_log_path = (
        Path(merge_static_log_path)
        if merge_static_log_path is not None
        else output_path.parent / DEFAULT_MERGE_STATIC_LOG_NAME
    )
    _report_input_merge_differences(
        merge_log,
        resolved_dynamic_log_path,
        resolved_static_log_path,
        len(paths),
    )
    write_output_pickle(output_data, output_path)

    ret_period_txt_path = (
        Path(return_period_txt_path)
        if return_period_txt_path is not None
        else return_period_txt_path_from_pickle(output_path)
    )
    write_return_period_txt(
        output_data,
        requested_station_ids,
        ret_period_txt_path,
        static_info_path=static_info_path,
    )

    print(f"Loaded {len(merged_data)} stations from {len(paths)} input file(s)")
    if edges is not None:
        print(f"Derived {len(requested_station_ids)} graph nodes from {len(edges)} edge(s)")
    print(f"Selected stations: {', '.join(requested_station_ids)}")
    if parsed_start_date is not None:
        print(f"Stored from: {_format_day(parsed_start_date)}")
    if parsed_end_date is not None:
        print(f"Stored until: {_format_day(parsed_end_date)}")
    print(f"Wrote {len(output_data)} stations to {output_path}")
    print(f"Wrote return-period txt to {ret_period_txt_path}")

    summaries = build_station_summaries(
        output_data,
        requested_station_ids,
        static_info_path=static_info_path,
    )
    common_interval = compute_common_interval(output_data, requested_station_ids)
    print_station_summaries(summaries, common_interval)
    return output_data


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data-processing CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Load pickle and/or semicolon-separated text station files, "
            "filter by station ids or graph edges, and write pickle and "
            "return-period txt outputs."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=None,
        help=(
            "One or more input files (.pkl/.pickle/.pckl or .txt/.csv). "
            f"Default: {DEFAULT_INPUT_PATHS}"
        ),
    )
    parser.add_argument(
        "-s",
        "--stations",
        nargs="+",
        default=None,
        help=(
            "Station ids (e.g. 061 170 018) or edges in the same argument "
            "(e.g. '061,170' '018,170' or '061->170'). "
            f"Default graph edges: {DEFAULT_EDGES}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output pickle path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--station-id",
        default=None,
        help="Optional station id override when loading a single text file.",
    )
    parser.add_argument(
        "--static-info",
        default=DEFAULT_STATIC_INFO_PATH,
        help=f"CSV with station names (default: {DEFAULT_STATIC_INFO_PATH}).",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=(
            "First day to store (YYYY-MM-DD). "
            f"Default: {DEFAULT_START_DATE}. Use 'none' to keep all days."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional last day to store (YYYY-MM-DD). Default: no upper limit.",
    )
    parser.add_argument(
        "--return-period-txt",
        default=None,
        help=(
            "Output txt path for returnperiod.py. "
            f"Default: same folder as pickle, named {DEFAULT_RETURN_PERIOD_TXT_NAME}"
        ),
    )
    parser.add_argument(
        "--merge-log",
        default=None,
        help=(
            "CSV path for dynamic differences found while merging input files. "
            f"Default: same folder as pickle, named {DEFAULT_MERGE_LOG_NAME}"
        ),
    )
    parser.add_argument(
        "--merge-static-log",
        default=None,
        help=(
            "CSV path for static differences found while merging input files. "
            f"Default: same folder as pickle, named {DEFAULT_MERGE_STATIC_LOG_NAME}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for loading, filtering, and exporting station data."""
    args = _parse_args()
    input_paths = args.input if args.input is not None else DEFAULT_INPUT_PATHS
    stations_or_edges = (
        args.stations if args.stations is not None else _edges_to_tokens(DEFAULT_EDGES)
    )
    if args.station_id and len(input_paths) != 1:
        raise ValueError("--station-id can only be used with a single text input file")
    start_date = None if str(args.start_date).lower() == "none" else args.start_date
    process_data(
        input_paths=input_paths,
        stations_or_edges=stations_or_edges,
        output_path=args.output,
        station_id_override=args.station_id,
        static_info_path=args.static_info,
        start_date=start_date,
        end_date=args.end_date,
        return_period_txt_path=args.return_period_txt,
        merge_log_path=args.merge_log,
        merge_static_log_path=args.merge_static_log,
    )


if __name__ == "__main__":
    main()
