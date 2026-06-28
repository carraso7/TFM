from __future__ import annotations

"""Build and export static catchment metadata for hydrological stations.

Reads station time-series pickles and raw SAIH text/CSV headers to assemble
a ``static_info.csv`` file with catchment descriptors, names, and coordinates.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import re
from typing import Any, Iterable

import pandas as pd


# DEFAULT_OUTPUT_CSV_PATH = "/mnt/d/streamflow_prediction/static_info.csv"
DEFAULT_OUTPUT_CSV_PATH = "/mnt/d/streamflow_prediction/static_info_canfranc_added.csv"
# DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_allstations_plus_static.pkl"
DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
DEFAULT_RAW_DATA_FOLDER = "/mnt/d/streamflow_prediction/00_datos_crudos_SAIH/00_datos_crudos_SAIH"

# DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_allstations_plus_static_checked.pkl"
# DEFAULT_OUTPUT_CSV_PATH = "/mnt/d/streamflow_prediction/static_info_checked.csv"

# DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_allstations_plus_static.pkl"
# DEFAULT_OUTPUT_CSV_PATH = "/mnt/d/streamflow_prediction/static_info_before_checked.csv"

STATIC_COLUMNS = {
	"Catchment area": "Catchment Area (km2)",
	"Elevation": "Elevation gauging station (m.a.s.l.)",
	"Agricultural area (%)": "Agricultural areas",
	"Forestal area (%)": "Forests",
	"Shrub area (%)": "Shrub and/or herbaceous vegetation",
}

OUTPUT_COLUMNS = [
	"Station name",
	"Catchment area",
	"Elevation",
	"Agricultural area (%)",
	"Forestal area (%)",
	"Shrub area (%)",
	"Latitude",
	"Longitude",
]


@dataclass(frozen=True)
class FallbackStationInfo:
	"""Manual metadata for stations missing from raw SAIH header files.

	Attributes:
		name: Human-readable station name.
		zone: UTM zone used with ``easting`` and ``northing``.
		easting: UTM easting coordinate in metres.
		northing: UTM northing coordinate in metres.
		elevation: Optional gauge elevation in metres above sea level.
	"""
	name: str
	zone: int
	easting: float
	northing: float
	elevation: float | None = None


DEFAULT_FALLBACK_STATIONS: dict[str, FallbackStationInfo] = {
	"271": FallbackStationInfo(
		name="Rio Aragon en Canfranc Antiguo",
		zone=30,
		easting=702637.6,
		northing=4732667.0,
		elevation=1045.0,
	),
	"062": FallbackStationInfo(
		name="Rio Veral en Binies",
		zone=30,
		easting=681462.8,
		northing=4724973.0,
		elevation=650.0,
	),
	"080": FallbackStationInfo(
		name="Rio Veral en Zuriza",
		zone=30,
		easting=678062.1,
		northing=4747863.8,
		elevation=1187.0,
	),
	"282": FallbackStationInfo(
		name="Rio Aragon en Martes",
		zone=30,
		easting=673704.9,
		northing=4717624.1,
		elevation=544.0,
	),
	"268": FallbackStationInfo(
		name="Rio Esca en Isaba",
		zone=30,
		easting=669316.2,
		northing=4747261.8,
		elevation=775.3,
	),
	"063": FallbackStationInfo(
		name="Rio Esca en Sigues",
		zone=30,
		easting=662955.1,
		northing=4723305.2,
		elevation=520.0,
	),
	"018": FallbackStationInfo(
		name="Rio Aragon en Jaca",
		zone=30,
		easting=700704.6,
		northing=4716908.0,
		elevation=770.0,
	),
	"061": FallbackStationInfo(
		name="Rio Subordan en Javierregay",
		zone=30,
		easting=684285.7,
		northing=4716269.0,
		elevation=628.0,
	),
	# "272": FallbackStationInfo(
	# 	name="Rio Aragon en Canfranc",
	# 	zone=30,
	# 	easting=702637.6,
	# 	northing=4732667.0,
	# 	elevation=1045.0,
	# ),
}


@dataclass
class StationHeaderInfo:
	"""Location and naming metadata parsed from a raw SAIH station file header.

	Attributes:
		station_id: Three-digit station identifier.
		latitude: Decimal latitude in degrees, if present in the header.
		longitude: Decimal longitude in degrees, if present in the header.
		name: Station name string, if present in the header.
	"""
	station_id: str
	latitude: float | None
	longitude: float | None
	name: str | None


def _parse_station_id_from_filename(path: Path) -> str | None:
	"""Extract a three-digit station id from a raw data filename.

	Args:
		path: Path to a raw SAIH ``.txt`` or ``.csv`` file.

	Returns:
		Last three digits of the first four-digit number in the stem, or ``None``.
	"""
	match = re.search(r"(\d{4})", path.stem)
	if not match:
		return None
	return match.group(1)[-3:]


def _normalize_station_id(value: str) -> str:
	"""Normalize a station id to a zero-padded three-character string.

	Args:
		value: Raw station id, optionally prefixed with ``A``.

	Returns:
		Uppercase three-digit station code (e.g. ``"018"``).
	"""
	value = value.strip().upper()
	if value.startswith("A"):
		value = value[1:]
	return value.zfill(3)


def _iter_data_files(raw_data_folder: Path) -> Iterable[Path]:
	"""Yield all ``.csv`` and ``.txt`` files under a raw data directory tree.

	Args:
		raw_data_folder: Root folder containing SAIH station export files.

	Yields:
		Paths to individual raw data files.
	"""
	for path in raw_data_folder.rglob("*"):
		if path.is_file() and path.suffix.lower() in {".csv", ".txt"}:
			yield path


def _parse_header_value(line: str) -> str | None:
	"""Return the value after the first colon in a header line.

	Args:
		line: Single line from a SAIH station file header.

	Returns:
		Stripped value to the right of ``:``, or ``None`` if no colon is present.
	"""
	if ":" not in line:
		return None
	return line.split(":", 1)[1].strip()


def _parse_float(value: str | None) -> float | None:
	"""Parse a European-style decimal string to float.

	Args:
		value: Numeric string, optionally using comma as decimal separator.

	Returns:
		Parsed float, or ``None`` if the value is empty or not numeric.
	"""
	if not value:
		return None
	try:
		return float(value.replace(",", "."))
	except ValueError:
		return None


def _utm_to_latlon(zone: int, easting: float, northing: float) -> tuple[float, float]:
	"""Convert WGS84 UTM coordinates to latitude and longitude.

	Assumes the northern hemisphere.

	Args:
		zone: UTM zone number (e.g. 30 for Spain).
		easting: UTM easting in metres.
		northing: UTM northing in metres.

	Returns:
		Tuple ``(latitude, longitude)`` in decimal degrees.
	"""
	# WGS84 UTM to lat/lon conversion (northern hemisphere).
	a = 6378137.0
	ecc_squared = 0.00669438
	k0 = 0.9996

	x = easting - 500000.0
	y = northing

	e1 = (1 - math.sqrt(1 - ecc_squared)) / (1 + math.sqrt(1 - ecc_squared))
	mu = y / (a * k0 * (1 - ecc_squared / 4 - 3 * ecc_squared**2 / 64 - 5 * ecc_squared**3 / 256))

	phi1 = (
		mu
		+ (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
		+ (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
		+ (151 * e1**3 / 96) * math.sin(6 * mu)
		+ (1097 * e1**4 / 512) * math.sin(8 * mu)
	)

	ecc_prime_squared = ecc_squared / (1 - ecc_squared)
	n1 = a / math.sqrt(1 - ecc_squared * math.sin(phi1) ** 2)
	t1 = math.tan(phi1) ** 2
	c1 = ecc_prime_squared * math.cos(phi1) ** 2
	r1 = a * (1 - ecc_squared) / (1 - ecc_squared * math.sin(phi1) ** 2) ** 1.5
	d = x / (n1 * k0)

	lat = (
		phi1
		- (n1 * math.tan(phi1) / r1)
		* (
			d**2 / 2
			- (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ecc_prime_squared) * d**4 / 24
			+ (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ecc_prime_squared - 3 * c1**2)
			* d**6
			/ 720
		)
	)

	lon = (
		d
		- (1 + 2 * t1 + c1) * d**3 / 6
		+ (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ecc_prime_squared + 24 * t1**2) * d**5 / 120
	)
	lon = math.radians((zone - 1) * 6 - 180 + 3) + lon / math.cos(phi1)

	return math.degrees(lat), math.degrees(lon)


def _parse_station_header(path: Path, station_id: str) -> StationHeaderInfo:
	"""Parse name and coordinates from the header of a raw SAIH station file.

	Args:
		path: Path to the semicolon-separated station export file.
		station_id: Normalized station identifier associated with the file.

	Returns:
		``StationHeaderInfo`` with any fields found before the data table.
	"""
	latitude = None
	longitude = None
	name = None

	with path.open("r", encoding="latin-1", errors="replace") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			line_lower = line.lower()

			if line_lower.startswith("serie de tiempo") and ";" in line_lower:
				break

			if "nombre de estaci" in line_lower:
				name = _parse_header_value(line)
				continue

			if "latitud" in line_lower:
				latitude = _parse_float(_parse_header_value(line))
				continue

			if "longitud" in line_lower:
				longitude = _parse_float(_parse_header_value(line))
				continue

	return StationHeaderInfo(
		station_id=station_id,
		latitude=latitude,
		longitude=longitude,
		name=name,
	)


def _extract_static_values(df: pd.DataFrame) -> dict[str, Any]:
	"""Extract time-invariant catchment attributes from the first data row.

	Args:
		df: Station DataFrame indexed by date with static columns repeated
			on every row.

	Returns:
		Dict keyed by ``OUTPUT_COLUMNS`` static names with scalar values from
		the first row, or ``None`` for missing columns when the frame is empty.
	"""
	if df.empty:
		return {key: None for key in STATIC_COLUMNS.keys()}
	first_row = df.iloc[0]
	values: dict[str, Any] = {}
	for output_col, source_col in STATIC_COLUMNS.items():
		values[output_col] = first_row.get(source_col)
	return values


def build_static_info_csv(
	raw_data_folder: str | Path,
	pickle_path: str | Path,
	output_csv_path: str | Path | None = None,
	fallback_stations: dict[str, FallbackStationInfo] | None = None,
) -> pd.DataFrame:
	"""Merge pickle static columns with raw header metadata and write CSV.

	Input pickle structure:
		``dict[str, pd.DataFrame]`` keyed by station id. Each DataFrame should
		contain the static columns mapped in ``STATIC_COLUMNS``.

	Output CSV structure:
		Indexed by ``station_id`` with columns in ``OUTPUT_COLUMNS``:
		station name, catchment area, elevation, land-cover percentages,
		latitude, and longitude.

	Args:
		raw_data_folder: Folder tree with SAIH ``.txt``/``.csv`` station exports.
		pickle_path: Pickle with per-station DataFrames including static fields.
		output_csv_path: Destination CSV path. Defaults to
			``raw_data_folder / "static_info.csv"``.
		fallback_stations: Optional map of station ids to manual coordinates
			and names used when raw headers are incomplete.

	Returns:
		Assembled static-info DataFrame written to ``output_csv_path``.
	"""
	raw_data_folder = Path(raw_data_folder)
	pickle_path = Path(pickle_path)
	if output_csv_path is None:
		output_csv_path = raw_data_folder / "static_info.csv"
	else:
		output_csv_path = Path(output_csv_path)

	with pickle_path.open("rb") as handle:
		data = pickle.load(handle)

	station_frames = {
		station_id: df for station_id, df in data.items() if isinstance(df, pd.DataFrame)
	}

	header_info: dict[str, StationHeaderInfo] = {}
	for path in _iter_data_files(raw_data_folder):
		station_id = _parse_station_id_from_filename(path)
		if not station_id:
			continue
		header_info[station_id] = _parse_station_header(path, station_id)

	pickle_station_ids = set(station_frames.keys())
	folder_station_ids = set(header_info.keys())

	only_in_folder = sorted(folder_station_ids - pickle_station_ids)
	only_in_pickle = sorted(pickle_station_ids - folder_station_ids)

	if only_in_folder:
		print(
			"Stations only in raw data folder: " + ", ".join(only_in_folder)
		)
	if only_in_pickle:
		print(
			"Stations only in pickle file: " + ", ".join(only_in_pickle)
		)

	rows: list[dict[str, Any]] = []
	fallback_stations = fallback_stations or DEFAULT_FALLBACK_STATIONS
	missing_lat_long: list[str] = []
	missing_raw_files: list[str] = []
	all_station_ids = sorted(pickle_station_ids | folder_station_ids)
	for station_id in all_station_ids:
		row = {"station_id": station_id}

		if station_id in station_frames:
			row.update(_extract_static_values(station_frames[station_id]))
		else:
			row.update({key: None for key in STATIC_COLUMNS.keys()})

		info = header_info.get(station_id)
		fallback = fallback_stations.get(_normalize_station_id(station_id))
		if info:
			row["Station name"] = info.name
			row["Latitude"] = info.latitude
			row["Longitude"] = info.longitude
			if info.latitude is None or info.longitude is None:
				if fallback:
					lat, lon = _utm_to_latlon(
						fallback.zone, fallback.easting, fallback.northing
					)
					row["Latitude"] = lat
					row["Longitude"] = lon
					if not row["Station name"]:
						row["Station name"] = fallback.name
				else:
					missing_lat_long.append(station_id)
		else:
			row["Station name"] = None
			row["Latitude"] = None
			row["Longitude"] = None
			if fallback:
				lat, lon = _utm_to_latlon(
					fallback.zone, fallback.easting, fallback.northing
				)
				row["Latitude"] = lat
				row["Longitude"] = lon
				row["Station name"] = fallback.name
			else:
				missing_raw_files.append(station_id)

		rows.append(row)

	output_df = pd.DataFrame(rows).set_index("station_id")
	output_df = output_df.reindex(columns=OUTPUT_COLUMNS)
	output_df.to_csv(output_csv_path)
	if missing_raw_files:
		print(
			"Stations missing raw data files: " + ", ".join(sorted(missing_raw_files))
		)
	if missing_lat_long:
		print(
			"Stations missing latitude/longitude: " + ", ".join(sorted(missing_lat_long))
		)
	print(f"Wrote static info CSV to {output_csv_path}")
	return output_df


def main() -> None:
	"""Build the default static-info CSV from configured pickle and raw paths."""
	build_static_info_csv(
		raw_data_folder=DEFAULT_RAW_DATA_FOLDER,
		pickle_path=DEFAULT_PICKLE_PATH,
		output_csv_path=DEFAULT_OUTPUT_CSV_PATH,
		fallback_stations=DEFAULT_FALLBACK_STATIONS,
	)


if __name__ == "__main__":
	main()
