
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from bokeh.models import Arrow, ColumnDataSource, HoverTool, LabelSet, NormalHead, WMTSTileSource
from bokeh.plotting import figure, output_file, save, show
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point


DEFAULT_STATIC_INFO_PATH = "/mnt/d/streamflow_prediction/static_info.csv"
DEFAULT_MAP_TILE_URL = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png"
DEFAULT_MAP_TILE_ATTRIBUTION = "&copy; OpenStreetMap contributors &copy; CARTO"
DEFAULT_QGIS_CRS = "EPSG:25830"
DEFAULT_ARROW_OFFSET_M = 2000.0
DEFAULT_RELATIONS = [("271", "018"), ("061", "282"), ("018", "282"), ("282", "101")] # Final de yesa
DEFAULT_RELATIONS = [("271", "018"), ("061", "282"), ("018", "282"), ("282", "170")] # Inicial de yesa
DEFAULT_RELATIONS = [("271", "018"), ("061", "282"), ("018", "282"), ("282", "170"), ("080", "062"), ("062", "170")] # Con todos los posibles nodos    , ("", "")
DEFAULT_RELATIONS = [("061", "170"), ("018", "170"),  ("080", "062"), ("062", "170")] # Con todos los posibles nodos sin los no procesados por conchi   , ("", "")
DEFAULT_RELATIONS: list[tuple[str, str]] = [ # Con todos los posibles nodos sin los no procesados por conchi y añadiendo canfranc
    ("061", "170"),
    ("018", "170"),
    ("080", "062"),
    ("062", "170"),
    ("271", "018"),
]
DEFAULT_STATION_IDS: list[str] | None = None
DEFAULT_VISUALS_DIR = Path(__file__).resolve().parent / "visuals"

def _build_name_to_id(static_info: pd.DataFrame | str | Path) -> dict[str, str]:
	if isinstance(static_info, (str, Path)):
		static_info_df = pd.read_csv(static_info, dtype={"station_id": str})
	else:
		static_info_df = static_info.copy()
	if "station_id" not in static_info_df.columns:
		static_info_df = static_info_df.reset_index()
	if "Station name" not in static_info_df.columns:
		raise ValueError("static_info must include a 'Station name' column")

	name_to_id: dict[str, str] = {}
	for _, row in static_info_df.iterrows():
		name = row.get("Station name")
		station_id = row.get("station_id")
		if pd.isna(name) or pd.isna(station_id):
			continue
		name_key = str(name).strip().lower()
		station_id_str = str(station_id)
		if name_key in name_to_id and name_to_id[name_key] != station_id_str:
			duplicate_key = f"{name_key} ({station_id_str})"
			if duplicate_key in name_to_id and name_to_id[duplicate_key] != station_id_str:
				raise ValueError(f"Duplicate station id mapping for {station_id}")
			name_to_id[duplicate_key] = station_id_str
		else:
			name_to_id[name_key] = station_id_str

	return name_to_id


def create_adj_matrix_hydrological(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
	name_to_id: dict[str, str] = {}
	if static_info is not None:
		name_to_id = _build_name_to_id(static_info)

	station_id_set = set(str(station_id) for station_id in station_ids or [])

	def _resolve_station_id(value: str) -> str:
		value_str = str(value)
		if station_id_set and value_str in station_id_set:
			return value_str

		name_key = value_str.strip().lower()
		if name_key in name_to_id:
			return name_to_id[name_key]
		if station_id_set:
			raise ValueError(f"Unknown station reference: {value}")
		return value_str
	graph = nx.DiGraph()
	if station_id_set:
		graph.add_nodes_from(sorted(station_id_set))
	for source, target in relations:
		src_id = _resolve_station_id(source)
		tgt_id = _resolve_station_id(target)
		graph.add_edge(src_id, tgt_id)

	node_order = sorted(graph.nodes())
	adj_matrix = nx.to_numpy_array(graph, nodelist=node_order, dtype=int)
	return pd.DataFrame(adj_matrix, index=node_order, columns=node_order)


def create_adj_matrix_dense(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
	base_matrix = create_adj_matrix_hydrological(station_ids, relations, static_info)
	node_order = list(base_matrix.index)
	n = len(node_order)
	adj_matrix = np.ones((n, n), dtype=int)
	np.fill_diagonal(adj_matrix, 0)
	return pd.DataFrame(adj_matrix, index=node_order, columns=node_order)


def create_adj_matrix_all_paths(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
	adj_matrix = create_adj_matrix_hydrological(station_ids, relations, static_info)
	n = len(adj_matrix)
	matrix_with_diag = adj_matrix.to_numpy(dtype=int).copy()
	np.fill_diagonal(matrix_with_diag, 1)
	powered = np.linalg.matrix_power(matrix_with_diag, n)
	all_paths = (powered != 0).astype(int)
	np.fill_diagonal(all_paths, 0)
	return pd.DataFrame(
		all_paths,
		index=adj_matrix.index,
		columns=adj_matrix.columns,
	)



def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	radius = 6371000.0
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	delta_phi = math.radians(lat2 - lat1)
	delta_lambda = math.radians(lon2 - lon1)

	a = (
		math.sin(delta_phi / 2) ** 2
		+ math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
	)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return radius * c


def _lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
	r_major = 6378137.0
	x = math.radians(lon) * r_major
	lat = max(min(lat, 89.9999), -89.9999)
	y = r_major * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
	return x, y


def _load_station_features(
	static_info_path: str | Path,
	station_ids: list[str],
) -> list[dict[str, object]]:
	static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
	static_info_df = static_info_df.set_index("station_id")

	stations: list[dict[str, object]] = []
	for station_id in station_ids:
		if station_id not in static_info_df.index:
			raise ValueError(f"Station {station_id} not found in static info")
		row = static_info_df.loc[station_id]
		lat = row.get("Latitude")
		lon = row.get("Longitude")
		if pd.isna(lat) or pd.isna(lon):
			raise ValueError(f"Missing latitude/longitude for station {station_id}")
		stations.append(
			{
				"station_id": station_id,
				"station_name": str(row.get("Station name") or ""),
				"lat": float(lat),
				"lon": float(lon),
				"catchment_area": (
					float(row.get("Catchment area"))
					if not pd.isna(row.get("Catchment area"))
					else None
				),
				"elevation": (
					float(row.get("Elevation"))
					if not pd.isna(row.get("Elevation"))
					else None
				),
			}
		)
	return stations


def _project_lonlat_to_crs(
	lon: float,
	lat: float,
	crs: str = DEFAULT_QGIS_CRS,
) -> tuple[float, float]:
	point_gdf = gpd.GeoDataFrame(
		geometry=[Point(lon, lat)],
		crs="EPSG:4326",
	)
	projected = point_gdf.to_crs(crs)
	point = projected.geometry.iloc[0]
	return float(point.x), float(point.y)


def _build_edge_features(
	station_ids: list[str],
	weighted_adj_matrix: pd.DataFrame,
	coord_lookup: dict[str, tuple[float, float]],
	arrow_offset_m: float = DEFAULT_ARROW_OFFSET_M,
) -> list[dict[str, object]]:
	edges: list[dict[str, object]] = []
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
				end_x = end_x - dx / dist * arrow_offset_m
				end_y = end_y - dy / dist * arrow_offset_m
			weight_value = float(weight)
			weight_km = weight_value / 1000.0
			edges.append(
				{
					"source_id": source_id,
					"target_id": target_id,
					"start_x": start_x,
					"start_y": start_y,
					"end_x": end_x,
					"end_y": end_y,
					"weight_m": weight_value,
					"weight_km": weight_km,
				}
			)
	return edges


def _to_geodataframes(
	stations: list[dict[str, object]],
	edges: list[dict[str, object]] | None,
	crs: str = DEFAULT_QGIS_CRS,
	node_extra_attributes: dict[str, dict[str, object]] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
	node_records: list[dict[str, object]] = []
	for station in stations:
		x, y = _project_lonlat_to_crs(station["lon"], station["lat"], crs=crs)
		record: dict[str, object] = {
			"station_id": station["station_id"],
			"station_name": station["station_name"],
			"lat": station["lat"],
			"lon": station["lon"],
			"catchment_area": station["catchment_area"],
			"elevation": station["elevation"],
			"geometry": Point(x, y),
		}
		if node_extra_attributes is not None:
			record.update(node_extra_attributes.get(station["station_id"], {}))
		node_records.append(record)

	nodes_gdf = gpd.GeoDataFrame(node_records, crs=crs)

	if not edges:
		return nodes_gdf, None

	edge_records: list[dict[str, object]] = []
	for edge in edges:
		line = LineString(
			[(edge["start_x"], edge["start_y"]), (edge["end_x"], edge["end_y"])]
		)
		edge_records.append(
			{
				"source_id": edge["source_id"],
				"target_id": edge["target_id"],
				"weight_m": edge["weight_m"],
				"weight_km": edge["weight_km"],
				"geometry": line,
			}
		)

	edges_gdf = gpd.GeoDataFrame(edge_records, crs=crs)
	return nodes_gdf, edges_gdf


def export_graph_for_qgis(
	output_dir: str | Path,
	gpkg_name: str,
	static_info_path: str | Path,
	station_ids: list[str],
	weighted_adj_matrix: pd.DataFrame | None = None,
	crs: str = DEFAULT_QGIS_CRS,
	node_extra_attributes: dict[str, dict[str, object]] | None = None,
) -> Path:
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	gpkg_path = output_path / gpkg_name

	stations = _load_station_features(static_info_path, station_ids)
	edges: list[dict[str, object]] | None = None
	if weighted_adj_matrix is not None:
		coord_lookup = {
			station["station_id"]: _project_lonlat_to_crs(
				station["lon"],
				station["lat"],
				crs=crs,
			)
			for station in stations
		}
		edges = _build_edge_features(
			station_ids,
			weighted_adj_matrix,
			coord_lookup,
		)

	nodes_gdf, edges_gdf = _to_geodataframes(
		stations,
		edges,
		crs=crs,
		node_extra_attributes=node_extra_attributes,
	)

	if gpkg_path.exists():
		gpkg_path.unlink()

	nodes_gdf.to_file(gpkg_path, layer="nodes", driver="GPKG")
	if edges_gdf is not None:
		edges_gdf.to_file(gpkg_path, layer="edges", driver="GPKG")

	return gpkg_path


def create_weighted_adj_matrix_distances_from_adj(
	adj_matrix: pd.DataFrame,
	static_info_path: str | Path,
) -> pd.DataFrame:
	if not isinstance(adj_matrix, pd.DataFrame):
		raise TypeError("adj_matrix must be a pandas DataFrame")
	if list(adj_matrix.index) != list(adj_matrix.columns):
		raise ValueError("adj_matrix index and columns must match")

	static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
	static_info_df = static_info_df.set_index("station_id")

	coords: dict[str, tuple[float, float]] = {}
	for station_id in adj_matrix.index:
		if station_id not in static_info_df.index:
			raise ValueError(f"Station {station_id} not found in static info")
		row = static_info_df.loc[station_id]
		lat = row.get("Latitude")
		lon = row.get("Longitude")
		if pd.isna(lat) or pd.isna(lon):
			raise ValueError(f"Missing latitude/longitude for station {station_id}")
		coords[station_id] = (float(lat), float(lon))

	weighted = pd.DataFrame(
		0.0,
		index=adj_matrix.index,
		columns=adj_matrix.columns,
	)
	for source_id in adj_matrix.index:
		for target_id in adj_matrix.columns:
			if adj_matrix.at[source_id, target_id] == 0:
				continue
			lat1, lon1 = coords[source_id]
			lat2, lon2 = coords[target_id]
			weighted.at[source_id, target_id] = _haversine_meters(lat1, lon1, lat2, lon2)

	return weighted


def create_weighted_adj_matrix_all_river_distances_from_hydrological(
	adj_matrix: pd.DataFrame,
	static_info_path: str | Path,
) -> pd.DataFrame:
	if not isinstance(adj_matrix, pd.DataFrame):
		raise TypeError("hydrological_adj_matrix must be a pandas DataFrame")
	if list(adj_matrix.index) != list(adj_matrix.columns):
		raise ValueError("hydrological_adj_matrix index and columns must match")

	static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
	static_info_df = static_info_df.set_index("station_id")

	coords: dict[str, tuple[float, float]] = {}
	for station_id in adj_matrix.index:
		if station_id not in static_info_df.index:
			raise ValueError(f"Station {station_id} not found in static info")
		row = static_info_df.loc[station_id]
		lat = row.get("Latitude")
		lon = row.get("Longitude")
		if pd.isna(lat) or pd.isna(lon):
			raise ValueError(f"Missing latitude/longitude for station {station_id}")
		coords[station_id] = (float(lat), float(lon))

	node_order = list(adj_matrix.index)
	n = len(node_order)

	def _geo_distance(source_id: str, target_id: str) -> float:
		lat1, lon1 = coords[source_id]
		lat2, lon2 = coords[target_id]
		return _haversine_meters(lat1, lon1, lat2, lon2)

	hydro_graph = nx.DiGraph()
	hydro_graph.add_nodes_from(node_order)
	for source_id in node_order:
		for target_id in node_order:
			if adj_matrix.at[source_id, target_id] == 0:
				continue
			hydro_graph.add_edge(
				source_id,
				target_id,
				weight=_geo_distance(source_id, target_id),
			)

	matrix_with_diag = adj_matrix.to_numpy(dtype=int).copy()
	np.fill_diagonal(matrix_with_diag, 1)
	powered = np.linalg.matrix_power(matrix_with_diag, n)
	all_paths = (powered != 0).astype(int)
	np.fill_diagonal(all_paths, 0)

	weighted = pd.DataFrame(
		0.0,
		index=node_order,
		columns=node_order,
	)
	for source_idx, source_id in enumerate(node_order):
		for target_idx, target_id in enumerate(node_order):
			if all_paths[source_idx, target_idx] == 0:
				continue
			if adj_matrix.at[source_id, target_id] != 0:
				weighted.at[source_id, target_id] = _geo_distance(source_id, target_id)
				continue

			path = nx.shortest_path(hydro_graph, source_id, target_id, weight="weight")
			path_weight = 0.0
			for hop_idx in range(len(path) - 1):
				path_weight += hydro_graph[path[hop_idx]][path[hop_idx + 1]]["weight"]
			weighted.at[source_id, target_id] = path_weight

	return weighted




def plot_weighted_graph_map(
	weighted_adj_matrix: pd.DataFrame,
	static_info_path: str | Path,
	output_html: str | Path | None = None,
	show_plot: bool = False,
	map_tile_url: str = DEFAULT_MAP_TILE_URL,
	map_tile_attribution: str = DEFAULT_MAP_TILE_ATTRIBUTION,
	output_qgis_dir: str | Path | None = None,
	qgis_gpkg_name: str = "weighted_graph.gpkg",
) -> None:
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
		}
	)

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
	arrow_offset = 2000.0

	coord_lookup = dict(zip(station_ids, zip(x_coords, y_coords)))
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

	plot = figure(
		title="Weighted station graph",
		x_axis_type="mercator",
		y_axis_type="mercator",
		x_axis_label="Longitude",
		y_axis_label="Latitude",
		width=900,
		height=700,
	)
	tile_source = WMTSTileSource(
		url=map_tile_url,
		attribution=map_tile_attribution,
	)
	plot.add_tile(tile_source)

	edge_renderer = plot.segment(
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
		color="#1f77b4",
		hover_color="#b91c1c",
	)

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
			("Catchment area", "@catchment_area{0.00}"),
			("Elevation", "@elevation{0.00}"),
			("Agricultural", "@agri_area{0.00}"),
			("Forestal", "@forest_area{0.00}"),
			("Shrub", "@shrub_area{0.00}"),
		],
	)
	plot.add_tools(node_hover)

	if output_html is not None:
		output_file(str(output_html))
		save(plot)
	if output_qgis_dir is not None:
		export_graph_for_qgis(
			output_dir=output_qgis_dir,
			gpkg_name=qgis_gpkg_name,
			static_info_path=static_info_path,
			station_ids=station_ids,
			weighted_adj_matrix=weighted_adj_matrix,
		)
	if show_plot:
		show(plot)


def show_only_nodes(
	station_ids: list[str],
	static_info_path: str | Path,
	output_html: str | Path | None = None,
	show_plot: bool = False,
	map_tile_url: str = DEFAULT_MAP_TILE_URL,
	map_tile_attribution: str = DEFAULT_MAP_TILE_ATTRIBUTION,
	output_qgis_dir: str | Path | None = None,
	qgis_gpkg_name: str = "nodes_only.gpkg",
) -> None:
	static_info_df = pd.read_csv(static_info_path, dtype={"station_id": str})
	static_info_df = static_info_df.set_index("station_id")

	latitudes: list[float] = []
	longitudes: list[float] = []
	x_coords: list[float] = []
	y_coords: list[float] = []
	station_names: list[str] = []
	label_texts: list[str] = []
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
		name = str(row.get("Station name") or "")
		if " en " in name:
			left, right = name.split(" en ", 1)
			name = f"{left} en\n{right}"
		station_names.append(name)
		label_texts.append(f"{station_id}\n{name}")

	nodes_source = ColumnDataSource(
		{
			"station_id": station_ids,
			"station_name": station_names,
			"lat": latitudes,
			"lon": longitudes,
			"x": x_coords,
			"y": y_coords,
			"label": label_texts,
		}
	)

	plot = figure(
		title="Stations (nodes only)",
		x_axis_type="mercator",
		y_axis_type="mercator",
		x_axis_label="Longitude",
		y_axis_label="Latitude",
		width=900,
		height=700,
	)
	tile_source = WMTSTileSource(
		url=map_tile_url,
		attribution=map_tile_attribution,
	)
	plot.add_tile(tile_source)
	plot.circle(
		"x",
		"y",
		size=16,
		source=nodes_source,
		color="#1f77b4",
	)

	node_hover = HoverTool(
		tooltips=[
			("Station", "@station_id"),
			("Name", "@station_name"),
			("Latitude", "@lat{0.0000}"),
			("Longitude", "@lon{0.0000}"),
		],
	)
	plot.add_tools(node_hover)

	node_labels = LabelSet(
		x="x",
		y="y",
		text="label",
		source=nodes_source,
		x_offset=6,
		y_offset=6,
		text_color="#000000",
		level="overlay",
	)
	plot.add_layout(node_labels)

	if output_html is not None:
		output_file(str(output_html))
		save(plot)
	if output_qgis_dir is not None:
		export_graph_for_qgis(
			output_dir=output_qgis_dir,
			gpkg_name=qgis_gpkg_name,
			static_info_path=static_info_path,
			station_ids=station_ids,
		)
	if show_plot:
		show(plot)


def create_weighted_adj_matrix_all_river_distances(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
	verbose: int = 0,
) -> pd.DataFrame:
	adj_matrix = create_adj_matrix_hydrological(station_ids, relations, static_info)
	if verbose > 0:
		print(adj_matrix)
	return create_weighted_adj_matrix_all_river_distances_from_hydrological(adj_matrix, static_info)


def create_weighted_adj_matrix_all_paths(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
	verbose: int = 0,
) -> pd.DataFrame:
	adj_matrix = create_adj_matrix_all_paths(station_ids, relations, static_info)
	if verbose > 0:
		print(adj_matrix)
	return create_weighted_adj_matrix_distances_from_adj(adj_matrix, static_info)


def create_weighted_adj_matrix_hydrological(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
	verbose: int = 0,
) -> pd.DataFrame:
	adj_matrix = create_adj_matrix_hydrological(station_ids, relations, static_info)
	if verbose > 0:
		print(adj_matrix)
	return create_weighted_adj_matrix_distances_from_adj(adj_matrix, static_info)


def create_weighted_adj_matrix_dense(
	station_ids: list[str] | None,
	relations: list[tuple[str, str]],
	static_info: pd.DataFrame | str | Path | None = None,
	verbose: int = 0,
) -> pd.DataFrame:
	adj_matrix = create_adj_matrix_dense(station_ids, relations, static_info)
	if verbose > 0:
		print(adj_matrix)
	return create_weighted_adj_matrix_distances_from_adj(adj_matrix, static_info)



create_weighted_adj_matrix = create_weighted_adj_matrix_hydrological

def main() -> None:
	DEFAULT_VISUALS_DIR.mkdir(parents=True, exist_ok=True)
	weighted_adj_matrix = create_weighted_adj_matrix(
		station_ids=DEFAULT_STATION_IDS,
		relations=DEFAULT_RELATIONS,
		static_info=DEFAULT_STATIC_INFO_PATH,
		verbose=0,
	)
	plot_weighted_graph_map(
		weighted_adj_matrix=weighted_adj_matrix,
		static_info_path=DEFAULT_STATIC_INFO_PATH,
		output_html=DEFAULT_VISUALS_DIR / "weighted_graph.html",
		output_qgis_dir=DEFAULT_VISUALS_DIR / "for_QGIS" / "weighted_graph",
		show_plot=False,
	)
	show_only_nodes(
		station_ids=list(weighted_adj_matrix.index),
		static_info_path=DEFAULT_STATIC_INFO_PATH,
		output_html=DEFAULT_VISUALS_DIR / "nodes_only_graph.html",
		output_qgis_dir=DEFAULT_VISUALS_DIR / "for_QGIS" / "nodes_only",
		show_plot=False,
	)
	print(weighted_adj_matrix)


if __name__ == "__main__":
	main()
