#!/usr/bin/env python3
from __future__ import annotations

from pathlib import PureWindowsPath
import pickle
from typing import Any

import pandas as pd


WINDOWS_PICKLE_PATH = r"D:\streamflow_prediction\inputs_allstations.pkl"
WINDOWS_PICKLE_PATH_2 = r"D:\streamflow_prediction\inputs_allstations_plus_static.pkl"
WINDOWS_PICKLE_PATH_2 = r"D:\streamflow_prediction\inputs_allstations_plus_static_checked.pkl"

EXPECTED_COLUMNS = [
    "Streamflow [m3/s]",
    "Catchment-averaged precipitation [mm]",
    "Maximum temperature [°C]",
    "Minimum temperature [°C]",
    "Air humidity [%]",
    "SPEI [−]",
    "NAO [−]",
    "WeMO [−]",
]
EXPECTED_START = pd.Timestamp("1989-10-01")
EXPECTED_END = pd.Timestamp("2022-09-30")


def windows_to_wsl_path(path: str) -> str:
    """Convert a Windows path like D:\\folder\\file.pkl to /mnt/d/folder/file.pkl."""
    if len(path) >= 3 and path[1:3] == ":\\":
        p = PureWindowsPath(path)
        drive = p.drive[0].lower()
        tail = "/".join(p.parts[1:])
        return f"/mnt/{drive}/{tail}"
    return path


def preview_object(data: Any) -> None:
    print(f"Loaded object type: {type(data)}")

    # pandas-like objects
    if hasattr(data, "shape"):
        print(f"Shape: {getattr(data, 'shape', None)}")
    if hasattr(data, "columns"):
        cols = list(getattr(data, "columns"))
        print(f"Columns ({len(cols)}): {cols[:15]}{' ...' if len(cols) > 15 else ''}")
    if hasattr(data, "head"):
        print("\nHead (first 5 rows):")
        try:
            print(data.head())
            return
        except Exception:
            pass

    # dict-like
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Dictionary keys ({len(keys)}): {keys[:15]}{' ...' if len(keys) > 15 else ''}")
        if keys:
            first_key = keys[0]
            print(f"\nSample value for key {first_key!r}: {type(data[first_key])}")
        return

    # list/tuple-like
    if isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        print("First 3 items:")
        for i, item in enumerate(data[:3], start=1):
            print(f"  {i}. {type(item)} -> {repr(item)[:200]}")
        return

    print("\nPreview:")
    print(repr(data)[:1000])


def summarize_station_dictionary(data: dict[str, Any]) -> None:
    station_ids = sorted(data.keys())
    print(f"\nStations: {len(station_ids)}")
    print(f"First station IDs: {station_ids[:10]}{' ...' if len(station_ids) > 10 else ''}")

    check_ids = ["101", "282", "018", "271", "061", "062", "080", "063", "068", "170"]
    presence = {station_id: (station_id in data) for station_id in check_ids}
    print("\nRequested station IDs present:")
    for station_id in check_ids:
        print(f"- {station_id}: {presence[station_id]}")

    dataframe_stations = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
    print(f"Stations with DataFrame payload: {len(dataframe_stations)}/{len(data)}")
    if not dataframe_stations:
        return

    sample_station = sorted(dataframe_stations.keys())[0]
    sample_df = dataframe_stations[sample_station]

    print("\nExpected column order from README:")
    for i, col in enumerate(EXPECTED_COLUMNS, start=1):
        print(f"  {i}. {col}")

    print(f"\nSample station: {sample_station}")
    print(f"Shape: {sample_df.shape}")
    print(f"Index range: {sample_df.index.min()} -> {sample_df.index.max()}")
    print(f"Columns: {list(sample_df.columns)}")

    # Consistency checks across all stations.
    full_range_ok = 0
    columns_ok = 0
    leap_day_present = 0

    row_counts: list[int] = []
    for df in dataframe_stations.values():
        row_counts.append(len(df))

        if len(df.index) > 0 and df.index.min() == EXPECTED_START and df.index.max() == EXPECTED_END:
            full_range_ok += 1

        if list(df.columns) == EXPECTED_COLUMNS:
            columns_ok += 1

        if isinstance(df.index, pd.DatetimeIndex):
            has_leap = ((df.index.month == 2) & (df.index.day == 29)).any()
            if has_leap:
                leap_day_present += 1

    print("\nDataset checks:")
    print(f"- Date range matches {EXPECTED_START.date()}..{EXPECTED_END.date()}: {full_range_ok}/{len(dataframe_stations)} stations")
    print(f"- Column order exactly matches README: {columns_ok}/{len(dataframe_stations)} stations")
    print(f"- Stations containing 29/02 records: {leap_day_present}/{len(dataframe_stations)}")
    print(f"- Row count min/median/max: {min(row_counts)}/{int(pd.Series(row_counts).median())}/{max(row_counts)}")

    # Missing-value summary on sample station plus overall estimate.
    sample_missing_pct = (sample_df.isna().mean() * 100).round(2)
    print("\nMissing values in sample station (%):")
    for col, pct in sample_missing_pct.items():
        print(f"- {col}: {pct}%")

    concat_df = pd.concat(dataframe_stations.values(), axis=0, ignore_index=True)
    overall_missing_pct = (concat_df.isna().mean() * 100).round(2)
    print("\nOverall missing values across all stations (%):")
    for col, pct in overall_missing_pct.items():
        print(f"- {col}: {pct}%")

    print("\nSample station head (first 5 rows):")
    print(sample_df.head())


def main() -> None:
    pickle_path = windows_to_wsl_path(WINDOWS_PICKLE_PATH)
    print(f"Opening pickle file: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    preview_object(data)
    if isinstance(data, dict):
        summarize_station_dictionary(data)

    pickle_path = windows_to_wsl_path(WINDOWS_PICKLE_PATH_2)
    print(f"Opening pickle file: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    station_ids = sorted(data.keys())
    print(f"\nStations: {len(station_ids)}")
    print(f"First station IDs: {station_ids[:10]}{' ...' if len(station_ids) > 10 else ''}")


    dataframe_stations = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
    sample_station = sorted(dataframe_stations.keys())[0]
    sample_df = dataframe_stations[sample_station]

    print(sample_df)

    print(f"\nSample station: {sample_station}")
    print(f"Shape: {sample_df.shape}")
    print(f"Index range: {sample_df.index.min()} -> {sample_df.index.max()}")
    print(f"Columns: {list(sample_df.columns)}")


if __name__ == "__main__":
    main()
