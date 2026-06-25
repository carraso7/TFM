#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pickle
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from visuals import (
    ERROR_METRICS,
    comparison_NSE_conchi,
    load_station_name_map,
    compute_station_error,
    plot_graph_error_map,
    plot_KGE_separated_map,
    plot_random_year_predictions,
    plot_return_period_nrmse_boxplots,
    plot_return_period_nrmse_lineplots,
    plot_test_years_predictions,
    collect_return_period_nrmse_values,
)

from adj_matrix_visualize_maps_GNNs import (
    DEFAULT_RELATIONS,
    DEFAULT_STATIC_INFO_PATH,
    DEFAULT_STATION_IDS,
    create_weighted_adj_matrix_all_paths,
    create_weighted_adj_matrix_all_river_distances,
    create_weighted_adj_matrix_dense,
    create_weighted_adj_matrix_hydrological,
)


### ID 2
DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals"
DEFAULT_MODEL_DIR = "/mnt/d/streamflow_prediction/models"
WINDOW_DAYS = 365 # 1460 # 365
BATCH_SIZE = 16 # 32
EPOCHS = 120 # 30 # 120 # 60 # 30
LR = 1e-4
TEST_FRACTION = 0.2
SEED = 42
HIDDEN_DIM = 128 # 64 #256 # 64 cambiado para la segunda vez
MESSAGE_PASSES = 12 # 3
TRAINING_METRIC = "RMSE" # "MSE"
_LOSS_EPS = 1e-8
AGGREGATION = "max" # "sum"
DEFAULT_SEVERAL_GNN_LAYERS = True
UNDIRECTED_GRAPH = False
SELF_LOOPS = True
TRAIN_MESSAGE_PASSING = False
DEFAULT_ADJ_NORMALIZATION = "inv_dist"

### ACTUAL BASELINE:
DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals"
DEFAULT_MODEL_DIR = "/mnt/d/streamflow_prediction/models"
WINDOW_DAYS = 1460 # 365
BATCH_SIZE = 16 # 32
EPOCHS = 60 # 30 # 120 # 60 # 30
LR = 1e-3
TEST_FRACTION = 0.2
SEED = 42
HIDDEN_DIM = 64 # 64 #256 # 64 cambiado para la segunda vez
MESSAGE_PASSES = 3 # 3
TRAINING_METRIC = "RMSE" # "MSE"
_LOSS_EPS = 1e-8
AGGREGATION = "max" # "sum"
DEFAULT_SEVERAL_GNN_LAYERS = False
UNDIRECTED_GRAPH = False
SELF_LOOPS = True
TRAIN_MESSAGE_PASSING = True
DEFAULT_ADJ_NORMALIZATION = "inv_dist"

# ### FAST TRAINING
# DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
# DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals"
# DEFAULT_MODEL_DIR = "/mnt/d/streamflow_prediction/models"
# WINDOW_DAYS = 1460 # 365
# BATCH_SIZE = 16 # 32
# EPOCHS = 3 # 30 # 120 # 60 # 30
# LR = 1e-4
# TEST_FRACTION = 0.2
# SEED = 42
# HIDDEN_DIM = 64 # 64 #256 # 64 cambiado para la segunda vez
# MESSAGE_PASSES = 3 # 3
# TRAINING_METRIC = "RMSE" # "MSE"
# _LOSS_EPS = 1e-8
# AGGREGATION = "max" # "sum"
# DEFAULT_SEVERAL_GNN_LAYERS = False
# UNDIRECTED_GRAPH = False
# SELF_LOOPS = True
# TRAIN_MESSAGE_PASSING = False
# DEFAULT_ADJ_NORMALIZATION = "inv_dist"


DYNAMIC_COLUMNS = [
    "pr",
    "tmax_total",
    "tmin_total",
    "Humidity",
    "SPEI",
    "nao",
    "WEMO",
]
STATIC_COLUMNS = [
    "Catchment Area (km2)",
    "Elevation gauging station (m.a.s.l.)",
    "Agricultural areas",
    "Forests",
    "Shrub and/or herbaceous vegetation",
]
TARGET_COLUMN = "Streamflow"


def _normalize_adj_row_norm(adj: torch.Tensor, self_loops: bool = SELF_LOOPS, undirected: bool = UNDIRECTED_GRAPH) -> torch.Tensor:
    adj = adj.T.clone() # Transpose the adjacency matrix to make it column-wise and so that neighbours are upstream gauging stations.
    if undirected:
        adj = torch.maximum(adj, adj.T)  # Get bidirectional adjacency matrix.
    if self_loops:
        adj.fill_diagonal_(1.0)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


def _normalize_adj_inv_dist(adj: torch.Tensor, self_loops: bool = SELF_LOOPS, undirected: bool = UNDIRECTED_GRAPH) -> torch.Tensor:
    """Add self-loops if self_loops is True and convert non-zero edge weights to inverse distance (1/value)."""
    adj = adj.T.clone() # Transpose the adjacency matrix to make it column-wise and so that neighbours are upstream gauging stations.
    if undirected:
        adj = torch.maximum(adj, adj.T)  # Get bidirectional adjacency matrix.
    if self_loops:
        adj.fill_diagonal_(1.0)
    nonzero = adj != 0
    adj[nonzero] = 1.0 / adj[nonzero]
    return adj

DEFAULT_NORMALIZE_ADJ = _normalize_adj_inv_dist

VALID_ADJ_NORMALIZATIONS = ("row_norm", "inv_dist")
_ADJ_NORMALIZATION_FNS = {
    "row_norm": _normalize_adj_row_norm,
    "inv_dist": _normalize_adj_inv_dist,
}

DEFAULT_WEIGHTED_ADJ_FN = create_weighted_adj_matrix_all_river_distances

VALID_AGGREGATIONS = ("sum", "max", "mean")
VALID_TRAINING_METRICS = ("MSE", "RMSE")

WEIGHTED_ADJ_FN_BY_NAME: dict[str, Callable[..., pd.DataFrame]] = {
    "create_weighted_adj_matrix_hydrological": create_weighted_adj_matrix_hydrological,
    "create_weighted_adj_matrix_dense": create_weighted_adj_matrix_dense,
    "create_weighted_adj_matrix_all_paths": create_weighted_adj_matrix_all_paths,
    "create_weighted_adj_matrix_all_river_distances": create_weighted_adj_matrix_all_river_distances,
}

WEIGHTED_ADJ_ALIAS_BY_NAME: dict[str, str] = {
    "create_weighted_adj_matrix_hydrological": "hydro",
    "create_weighted_adj_matrix_dense": "dense",
    "create_weighted_adj_matrix_all_paths": "all_paths",
    "create_weighted_adj_matrix_all_river_distances": "river_dist",
}

WEIGHTED_ADJ_FN_BY_ALIAS: dict[str, Callable[..., pd.DataFrame]] = {
    alias: WEIGHTED_ADJ_FN_BY_NAME[name]
    for name, alias in WEIGHTED_ADJ_ALIAS_BY_NAME.items()
}

ALL_WEIGHTED_ADJ_FNS: list[Callable[..., pd.DataFrame]] = [
    create_weighted_adj_matrix_all_river_distances,
    create_weighted_adj_matrix_all_paths,
    create_weighted_adj_matrix_hydrological,
    create_weighted_adj_matrix_dense,
]
NUM_ADJ_STRATEGIES = len(ALL_WEIGHTED_ADJ_FNS)
VALID_MODEL_TYPES = ("dual", "mono")
DEFAULT_MODEL_TYPE = "dual"
TemporalModel = nn.Module


def _validate_aggregation(aggregation: str) -> str:
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {VALID_AGGREGATIONS}, got {aggregation!r}")
    return aggregation


def _resolve_weighted_adj_fn(name_or_alias: str | None) -> Callable[..., pd.DataFrame]:
    if name_or_alias is None:
        return DEFAULT_WEIGHTED_ADJ_FN
    if name_or_alias in WEIGHTED_ADJ_FN_BY_ALIAS:
        return WEIGHTED_ADJ_FN_BY_ALIAS[name_or_alias]
    if name_or_alias in WEIGHTED_ADJ_FN_BY_NAME:
        return WEIGHTED_ADJ_FN_BY_NAME[name_or_alias]
    raise ValueError(f"Unknown weighted adjacency function: {name_or_alias!r}")


def _validate_model_type(model_type: str) -> str:
    resolved = model_type.strip().lower()
    if resolved not in VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got {model_type!r}")
    return resolved


def _validate_adj_normalization(adj_normalization: str) -> str:
    resolved = adj_normalization.strip().lower()
    if resolved not in VALID_ADJ_NORMALIZATIONS:
        raise ValueError(
            f"adj_normalization must be one of {VALID_ADJ_NORMALIZATIONS}, got {adj_normalization!r}"
        )
    return resolved


def _make_normalize_adj_fn(
    adj_normalization: str,
    *,
    self_loops: bool = SELF_LOOPS,
    undirected: bool = UNDIRECTED_GRAPH,
) -> Callable[[torch.Tensor], torch.Tensor]:
    base_fn = _ADJ_NORMALIZATION_FNS[_validate_adj_normalization(adj_normalization)]

    def normalize(adj: torch.Tensor) -> torch.Tensor:
        return base_fn(adj, self_loops=self_loops, undirected=undirected)

    normalize.__name__ = base_fn.__name__
    return normalize


def is_mono_model_type(model_type: str) -> bool:
    return _validate_model_type(model_type) == "mono"


def _aggregate_neighbors(
    adj: torch.Tensor,
    h: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    aggregation = _validate_aggregation(aggregation)
    if aggregation == "sum":
        # H[b]^(l+1) = H[b]^(l) + ReLU(A_hat @ H[b]^(l))
        # H[b]^(l) in R^{NxH}: node embeddings for batch b at message pass l
        # A_hat in R^{NxN}: row-normalised adjacency matrix
        # einsum "ij,bjh->bih": i=target node, j=neighbour (summed), b=batch, h=hidden dim
        return torch.einsum("ij,bjh->bih", adj, h)
    if aggregation == "max":
        edge_mask = (adj != 0).unsqueeze(0).unsqueeze(-1)
        neighbor_h = h.unsqueeze(1)
        masked = torch.where(
            edge_mask,
            neighbor_h,
            torch.full_like(neighbor_h, float("-inf")),
        )
        return masked.max(dim=2).values
    row_sum = adj.sum(dim=1, keepdim=True)
    norm_adj = torch.where(row_sum > 0, adj / row_sum.clamp(min=_LOSS_EPS), torch.zeros_like(adj))
    return torch.einsum("ij,bjh->bih", norm_adj, h)


class MessagePassingStack(nn.Module):
    """Neighbour aggregation with optional learnable linear map and residual update."""

    def __init__(
        self,
        hidden_dim: int,
        message_passes: int,
        aggregation: str = AGGREGATION,
        train_message_passing: bool = TRAIN_MESSAGE_PASSING,
    ) -> None:
        super().__init__()
        self.message_passes = message_passes
        self.aggregation = _validate_aggregation(aggregation)
        self.train_message_passing = train_message_passing
        if train_message_passing:
            self.layers = nn.ModuleList(
                nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passes)
            )
        else:
            self.layers = nn.ModuleList()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.train_message_passing:
            for layer in self.layers:
                messages = _aggregate_neighbors(adj, h, self.aggregation)
                h = h + torch.relu(layer(messages))
            return h
        for _ in range(self.message_passes):
            h = _aggregate_neighbors(adj, h, self.aggregation)
        return h


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(SEED)


def _validate_training_metric(metric: str) -> str:
    canonical_metrics = {
        "mse": "MSE",
        "rmse": "RMSE",
    }
    resolved = canonical_metrics.get(metric.strip().lower())
    if resolved is None:
        raise ValueError(f"training_metric must be one of {VALID_TRAINING_METRICS}, got {metric!r}")
    return resolved


def _resolve_saved_training_metric(metric: str) -> str:
    try:
        return _validate_training_metric(metric)
    except ValueError:
        return TRAINING_METRIC


class TrainingMetricLoss(nn.Module):
    """Differentiable training loss for MSE and RMSE."""

    def __init__(self, metric: str = TRAINING_METRIC) -> None:
        super().__init__()
        self.metric = _validate_training_metric(metric)

    def forward(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        predicted = predicted.reshape(-1)
        observed = observed.reshape(-1)

        if self.metric == "MSE":
            return torch.mean((predicted - observed) ** 2)
        return torch.sqrt(torch.mean((predicted - observed) ** 2) + _LOSS_EPS)


@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    hidden_dim: int = HIDDEN_DIM
    lr: float = LR
    batch_size: int = BATCH_SIZE
    message_passes: int = MESSAGE_PASSES
    window_days: int = WINDOW_DAYS
    test_fraction: float = TEST_FRACTION
    seed: int = SEED
    training_metric: str = TRAINING_METRIC
    pickle_path: str | Path = DEFAULT_PICKLE_PATH
    static_info_path: str | Path = DEFAULT_STATIC_INFO_PATH
    model_dir: str | Path | None = DEFAULT_MODEL_DIR
    visuals_dir: str | Path | None = DEFAULT_VISUALS_DIR
    normalize_adj: Callable[[torch.Tensor], torch.Tensor] | None = None
    adj_normalization: str = DEFAULT_ADJ_NORMALIZATION
    undirected_graph: bool = UNDIRECTED_GRAPH
    self_loops: bool = SELF_LOOPS
    train_message_passing: bool = TRAIN_MESSAGE_PASSING
    weighted_adj_fn: Callable[..., pd.DataFrame] | None = None
    aggregation: str = "sum"
    model_type: str = DEFAULT_MODEL_TYPE
    several_gnn_layers: bool = DEFAULT_SEVERAL_GNN_LAYERS
    verbose: int = 2
    examine_train_test_peaks: bool = True
    run_name: str | None = None

    def normalize_adj_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.normalize_adj is not None:
            return self.normalize_adj
        return _make_normalize_adj_fn(
            self.adj_normalization,
            self_loops=self.self_loops,
            undirected=self.undirected_graph,
        )

    def resolve_weighted_adj_fn(self) -> Callable[..., pd.DataFrame]:
        return self.weighted_adj_fn or DEFAULT_WEIGHTED_ADJ_FN

    def uses_mono_encoder(self) -> bool:
        return is_mono_model_type(self.model_type)

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["pickle_path"] = str(self.pickle_path)
        data["static_info_path"] = str(self.static_info_path)
        if self.model_dir is not None:
            data["model_dir"] = str(self.model_dir)
        if self.visuals_dir is not None:
            data["visuals_dir"] = str(self.visuals_dir)
        data.pop("normalize_adj", None)
        data.pop("weighted_adj_fn", None)
        if self.normalize_adj is not None:
            data["normalize_adj"] = getattr(self.normalize_adj, "__name__", "custom")
        else:
            data["adj_normalization"] = _validate_adj_normalization(self.adj_normalization)
        data["undirected_graph"] = bool(self.undirected_graph)
        data["self_loops"] = bool(self.self_loops)
        data["train_message_passing"] = bool(self.train_message_passing)
        if self.several_gnn_layers:
            data["weighted_adj"] = "multi"
        else:
            weighted_fn = self.resolve_weighted_adj_fn()
            fn_name = getattr(weighted_fn, "__name__", "create_weighted_adj_matrix_hydrological")
            data["weighted_adj"] = WEIGHTED_ADJ_ALIAS_BY_NAME.get(fn_name, fn_name)
        data["aggregation"] = _validate_aggregation(self.aggregation)
        data["model_type"] = _validate_model_type(self.model_type)
        data["several_gnn_layers"] = bool(self.several_gnn_layers)
        data["training_metric"] = _validate_training_metric(self.training_metric)
        data["verbose"] = int(self.verbose)
        return data


@dataclass
class RunResult:
    config: TrainConfig
    model: nn.Module
    station_ids: list[str]
    series: list[StationSeries]
    split_idx: int
    test_start_idx: int
    test_loader: DataLoader
    weighted_adj: pd.DataFrame
    prediction_frames: dict[str, pd.DataFrame]
    errors_by_metric: dict[str, dict[str, float]]
    return_period_values: dict[float, dict[str, float]]


@dataclass
class StationSeries:
    station_id: str
    dynamic: np.ndarray
    target: np.ndarray
    static: np.ndarray
    dates: np.ndarray


def _iter_station_frames(data: dict) -> Iterable[tuple[str, pd.DataFrame]]:
    for station_id, df in data.items():
        if isinstance(df, pd.DataFrame):
            yield station_id, df


def _clamp_streamflow_predictions(preds: torch.Tensor) -> torch.Tensor:
    """Inference-only floor at zero; training still uses unconstrained model outputs."""
    return torch.clamp(preds, min=0.0)


def _broadcast_static_into_window(
    dynamic_window: np.ndarray,
    static_features: np.ndarray,
) -> np.ndarray:
    """Concatenate static features to every timestep in a node's dynamic window."""
    window_days = dynamic_window.shape[0]
    static_tiled = np.tile(static_features, (window_days, 1))
    return np.concatenate([dynamic_window, static_tiled], axis=-1)


class GraphWindowDataset(Dataset):
    def __init__(
        self,
        series: list[StationSeries],
        window_days: int,
        start_idx: int,
        end_idx: int,
    ) -> None:
        self.series = series
        self.window_days = window_days
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_nodes = len(series)
        self.dynamic_dim = series[0].dynamic.shape[1]
        self.static_dim = series[0].static.shape[0]

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = self.start_idx + idx + self.window_days
        start = end - self.window_days
        dynamic = np.zeros((self.num_nodes, self.window_days, self.dynamic_dim), dtype=np.float32)
        static = np.zeros((self.num_nodes, self.static_dim), dtype=np.float32)
        targets = np.zeros((self.num_nodes,), dtype=np.float32)
        for node_idx, station in enumerate(self.series):
            dynamic[node_idx] = station.dynamic[start:end]
            static[node_idx] = station.static
            targets[node_idx] = station.target[end - 1]
        return torch.from_numpy(dynamic), torch.from_numpy(static), torch.from_numpy(targets)


class GraphWindowDatasetLSTMOnly(Dataset):
    """Window dataset with static features repeated at every timestep in the LSTM input."""

    def __init__(
        self,
        series: list[StationSeries],
        window_days: int,
        start_idx: int,
        end_idx: int,
    ) -> None:
        self.series = series
        self.window_days = window_days
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_nodes = len(series)
        self.dynamic_dim = series[0].dynamic.shape[1]
        self.static_dim = series[0].static.shape[0]
        self.input_dim = self.dynamic_dim + self.static_dim

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.start_idx + idx + self.window_days
        start = end - self.window_days
        features = np.zeros((self.num_nodes, self.window_days, self.input_dim), dtype=np.float32)
        targets = np.zeros((self.num_nodes,), dtype=np.float32)
        for node_idx, station in enumerate(self.series):
            features[node_idx] = _broadcast_static_into_window(
                station.dynamic[start:end],
                station.static,
            )
            targets[node_idx] = station.target[end - 1]
        return torch.from_numpy(features), torch.from_numpy(targets)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, window_days, dynamic_dim = x.shape
        x = x.reshape(batch_size * num_nodes, window_days, dynamic_dim) # We treat each node's time series as a separate sequence in the batch for nn.LSTM structure simplicity. 
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1] # [0] Would also work since we have only 1 LSTM layer, but using [-1] is more robust if we later change to multiple layers.
        return h_last.reshape(batch_size, num_nodes, -1)


class StaticEncoder(nn.Module):
    def __init__(self, static_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalGNN(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int,
        message_passes: int,
        adj: torch.Tensor,
        normalize_adj: Callable[[torch.Tensor], torch.Tensor] | None = None,
        aggregation: str = AGGREGATION,
        train_message_passing: bool = TRAIN_MESSAGE_PASSING,
    ) -> None:
        super().__init__()
        self.temporal_encoder = LSTMEncoder(dynamic_dim, hidden_dim)
        self.static_encoder = StaticEncoder(static_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = MLPDecoder(hidden_dim)
        self.message_passing = MessagePassingStack(
            hidden_dim,
            message_passes,
            aggregation,
            train_message_passing=train_message_passing,
        )
        normalize_fn = normalize_adj or DEFAULT_NORMALIZE_ADJ
        self.register_buffer("adj", normalize_fn(adj))

    @property
    def message_passes(self) -> int:
        return self.message_passing.message_passes

    @property
    def aggregation(self) -> str:
        return self.message_passing.aggregation

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        temporal_h = self.temporal_encoder(dynamic)
        static_h = self.static_encoder(static)
        h = self.fusion(torch.cat([temporal_h, static_h], dim=-1))
        h = self.message_passing(h, self.adj)
        return self.decoder(h).squeeze(-1)


class TemporalGNNLSTMOnly(nn.Module):
    """Temporal GNN with a single LSTM encoder; static features are part of the input series."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        message_passes: int,
        adj: torch.Tensor,
        normalize_adj: Callable[[torch.Tensor], torch.Tensor] | None = None,
        aggregation: str = "sum",
        train_message_passing: bool = TRAIN_MESSAGE_PASSING,
    ) -> None:
        super().__init__()
        self.temporal_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.decoder = MLPDecoder(hidden_dim)
        self.message_passing = MessagePassingStack(
            hidden_dim,
            message_passes,
            aggregation,
            train_message_passing=train_message_passing,
        )
        normalize_fn = normalize_adj or DEFAULT_NORMALIZE_ADJ
        self.register_buffer("adj", normalize_fn(adj))

    @property
    def message_passes(self) -> int:
        return self.message_passing.message_passes

    @property
    def aggregation(self) -> str:
        return self.message_passing.aggregation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.temporal_encoder(features)
        h = self.message_passing(h, self.adj)
        return self.decoder(h).squeeze(-1)


class TemporalGNNMultiAdj(nn.Module):
    """Dual encoder with parallel message-passing branches, one per adjacency strategy."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        hidden_dim: int,
        message_passes: int,
        adjs: list[torch.Tensor],
        normalize_adj: Callable[[torch.Tensor], torch.Tensor] | None = None,
        aggregation: str = AGGREGATION,
        train_message_passing: bool = TRAIN_MESSAGE_PASSING,
    ) -> None:
        super().__init__()
        if not adjs:
            raise ValueError("TemporalGNNMultiAdj requires at least one adjacency matrix")
        self.temporal_encoder = LSTMEncoder(dynamic_dim, hidden_dim)
        self.static_encoder = StaticEncoder(static_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.num_branches = len(adjs)
        self.decoder = MLPDecoder(hidden_dim * self.num_branches, hidden_dim)
        self.message_passing = MessagePassingStack(
            hidden_dim,
            message_passes,
            aggregation,
            train_message_passing=train_message_passing,
        )
        normalize_fn = normalize_adj or DEFAULT_NORMALIZE_ADJ
        for idx, adj in enumerate(adjs):
            self.register_buffer(f"adj_{idx}", normalize_fn(adj))

    @property
    def message_passes(self) -> int:
        return self.message_passing.message_passes

    @property
    def aggregation(self) -> str:
        return self.message_passing.aggregation

    def branch_adjs(self) -> list[torch.Tensor]:
        return [getattr(self, f"adj_{idx}") for idx in range(self.num_branches)]

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        temporal_h = self.temporal_encoder(dynamic)
        static_h = self.static_encoder(static)
        h = self.fusion(torch.cat([temporal_h, static_h], dim=-1))
        branch_outputs = [
            self.message_passing(h, adj) for adj in self.branch_adjs()
        ]
        return self.decoder(torch.cat(branch_outputs, dim=-1)).squeeze(-1)


class TemporalGNNLSTMOnlyMultiAdj(nn.Module):
    """Mono encoder with parallel message-passing branches, one per adjacency strategy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        message_passes: int,
        adjs: list[torch.Tensor],
        normalize_adj: Callable[[torch.Tensor], torch.Tensor] | None = None,
        aggregation: str = "sum",
        train_message_passing: bool = TRAIN_MESSAGE_PASSING,
    ) -> None:
        super().__init__()
        if not adjs:
            raise ValueError("TemporalGNNLSTMOnlyMultiAdj requires at least one adjacency matrix")
        self.temporal_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.num_branches = len(adjs)
        self.decoder = MLPDecoder(hidden_dim * self.num_branches, hidden_dim)
        self.message_passing = MessagePassingStack(
            hidden_dim,
            message_passes,
            aggregation,
            train_message_passing=train_message_passing,
        )
        normalize_fn = normalize_adj or DEFAULT_NORMALIZE_ADJ
        for idx, adj in enumerate(adjs):
            self.register_buffer(f"adj_{idx}", normalize_fn(adj))

    @property
    def message_passes(self) -> int:
        return self.message_passing.message_passes

    @property
    def aggregation(self) -> str:
        return self.message_passing.aggregation

    def branch_adjs(self) -> list[torch.Tensor]:
        return [getattr(self, f"adj_{idx}") for idx in range(self.num_branches)]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.temporal_encoder(features)
        branch_outputs = [
            self.message_passing(h, adj) for adj in self.branch_adjs()
        ]
        return self.decoder(torch.cat(branch_outputs, dim=-1)).squeeze(-1)


def _is_mono_model(model: nn.Module) -> bool:
    return isinstance(model, (TemporalGNNLSTMOnly, TemporalGNNLSTMOnlyMultiAdj))


def build_temporal_model(
    config: TrainConfig,
    *,
    dynamic_dim: int,
    static_dim: int,
    input_dim: int,
    adj_tensor: torch.Tensor | None,
    adj_tensors: list[torch.Tensor] | None,
    device: torch.device,
) -> nn.Module:
    normalize_adj = config.normalize_adj_fn()
    aggregation = config.aggregation
    train_message_passing = config.train_message_passing
    mono = config.uses_mono_encoder()
    multi = config.several_gnn_layers

    if multi:
        if not adj_tensors:
            raise ValueError("several_gnn_layers=True requires adj_tensors")
        if mono:
            model: nn.Module = TemporalGNNLSTMOnlyMultiAdj(
                input_dim,
                config.hidden_dim,
                config.message_passes,
                adj_tensors,
                normalize_adj=normalize_adj,
                aggregation=aggregation,
                train_message_passing=train_message_passing,
            )
        else:
            model = TemporalGNNMultiAdj(
                dynamic_dim,
                static_dim,
                config.hidden_dim,
                config.message_passes,
                adj_tensors,
                normalize_adj=normalize_adj,
                aggregation=aggregation,
                train_message_passing=train_message_passing,
            )
    elif mono:
        if adj_tensor is None:
            raise ValueError("mono model requires adj_tensor")
        model = TemporalGNNLSTMOnly(
            input_dim,
            config.hidden_dim,
            config.message_passes,
            adj_tensor,
            normalize_adj=normalize_adj,
            aggregation=aggregation,
            train_message_passing=train_message_passing,
        )
    else:
        if adj_tensor is None:
            raise ValueError("dual model requires adj_tensor")
        model = TemporalGNN(
            dynamic_dim,
            static_dim,
            config.hidden_dim,
            config.message_passes,
            adj_tensor,
            normalize_adj=normalize_adj,
            aggregation=aggregation,
            train_message_passing=train_message_passing,
        )
    return model.to(device)


def print_temporal_gnn_summary(model: TemporalGNN, batch_size: int = 1, window_days: int = 365) -> None:
    """
    Print a structured summary of the TemporalGNN model.

    Displays layer names, output shapes (inferred from a dummy forward pass),
    parameter counts, and high-level hyperparameters stored on the model.

    Args:
        model (TemporalGNN): The GNN model to summarise.
        batch_size (int): Batch size used for the dummy forward pass (default 1).
        window_days (int): Sequence length fed to the LSTM encoder (default 365).
    """
    num_nodes   = model.adj.shape[0]
    dynamic_dim = model.temporal_encoder.lstm.input_size
    static_dim  = model.static_encoder.net[0].in_features
    hidden_dim  = model.temporal_encoder.lstm.hidden_size

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    sep = "=" * 62
    print(sep)
    print(f"  TemporalGNN — Model Summary")
    print(sep)
    print(f"  Nodes (stations)   : {num_nodes}")
    print(f"  Dynamic dim        : {dynamic_dim}")
    print(f"  Static dim         : {static_dim}")
    print(f"  Hidden dim         : {hidden_dim}")
    print(f"  Window days        : {window_days}")
    print(f"  Message-pass steps : {model.message_passes}")
    print(sep)
    print(f"  {'Component':<28} {'Output shape':<20} {'Params':>8}")
    print("-" * 62)

    device = next(model.parameters()).device

    dummy_dynamic = torch.zeros(batch_size, num_nodes, window_days, dynamic_dim, device=device)
    dummy_static  = torch.zeros(batch_size, num_nodes, static_dim, device=device)

    with torch.no_grad():
        # LSTMEncoder
        temporal_h = model.temporal_encoder(dummy_dynamic)
        lstm_params = sum(p.numel() for p in model.temporal_encoder.parameters())
        print(f"  {'LSTMEncoder':<28} {str(tuple(temporal_h.shape)):<20} {lstm_params:>8,}")

        # StaticEncoder
        static_h = model.static_encoder(dummy_static)
        static_params = sum(p.numel() for p in model.static_encoder.parameters())
        print(f"  {'StaticEncoder':<28} {str(tuple(static_h.shape)):<20} {static_params:>8,}")

        # Fusion
        h = model.fusion(torch.cat([temporal_h, static_h], dim=-1))
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        print(f"  {'Fusion (concat → Linear)':<28} {str(tuple(h.shape)):<20} {fusion_params:>8,}")

        # Message-passing (aggregate neighbours → optional linear → residual)
        if model.message_passing.train_message_passing:
            for i, layer in enumerate(model.message_passing.layers, start=1):
                messages = _aggregate_neighbors(model.adj, h, model.aggregation)
                h = h + torch.relu(layer(messages))
                layer_params = sum(p.numel() for p in layer.parameters())
                print(
                    f"  {'  GCN pass (linear+residual) ' + str(i):<28} "
                    f"{str(tuple(h.shape)):<20} {layer_params:>8,}"
                )
        else:
            for i in range(1, model.message_passes + 1):
                h = _aggregate_neighbors(model.adj, h, model.aggregation)
                print(
                    f"  {'  GCN pass (aggregate only) ' + str(i):<28} "
                    f"{str(tuple(h.shape)):<20} {0:>8,}"
                )

        # Decoder
        out = model.decoder(h)
        dec_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"  {'MLPDecoder':<28} {str(tuple(out.squeeze(-1).shape)):<20} {dec_params:>8,}")

    print(sep)
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Trainable params   : {trainable_params:,}")
    print(f"  Non-trainable      : {total_params - trainable_params:,}  (adj matrix buffer)")
    print(sep)


def print_temporal_gnn_lstm_only_summary(
    model: TemporalGNNLSTMOnly,
    batch_size: int = 1,
    window_days: int = 365,
) -> None:
    num_nodes = model.adj.shape[0]
    input_dim = model.temporal_encoder.lstm.input_size
    hidden_dim = model.temporal_encoder.lstm.hidden_size

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    sep = "=" * 62
    print(sep)
    print("  TemporalGNNLSTMOnly — Model Summary")
    print(sep)
    print(f"  Nodes (stations)   : {num_nodes}")
    print(f"  Input dim          : {input_dim}  (dynamic + static per timestep)")
    print(f"  Hidden dim         : {hidden_dim}")
    print(f"  Window days        : {window_days}")
    print(f"  Message-pass steps : {model.message_passes}")
    print(sep)
    print(f"  {'Component':<28} {'Output shape':<20} {'Params':>8}")
    print("-" * 62)

    device = next(model.parameters()).device
    dummy_features = torch.zeros(batch_size, num_nodes, window_days, input_dim, device=device)

    with torch.no_grad():
        h = model.temporal_encoder(dummy_features)
        lstm_params = sum(p.numel() for p in model.temporal_encoder.parameters())
        print(f"  {'LSTMEncoder':<28} {str(tuple(h.shape)):<20} {lstm_params:>8,}")

        if model.message_passing.train_message_passing:
            for i, layer in enumerate(model.message_passing.layers, start=1):
                messages = _aggregate_neighbors(model.adj, h, model.aggregation)
                h = h + torch.relu(layer(messages))
                layer_params = sum(p.numel() for p in layer.parameters())
                print(
                    f"  {'  GCN pass (linear+residual) ' + str(i):<28} "
                    f"{str(tuple(h.shape)):<20} {layer_params:>8,}"
                )
        else:
            for i in range(1, model.message_passes + 1):
                h = _aggregate_neighbors(model.adj, h, model.aggregation)
                print(
                    f"  {'  GCN pass (aggregate only) ' + str(i):<28} "
                    f"{str(tuple(h.shape)):<20} {0:>8,}"
                )

        out = model.decoder(h)
        dec_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"  {'MLPDecoder':<28} {str(tuple(out.squeeze(-1).shape)):<20} {dec_params:>8,}")

    print(sep)
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Trainable params   : {trainable_params:,}")
    print(f"  Non-trainable      : {total_params - trainable_params:,}  (adj matrix buffer)")
    print(sep)


def load_station_series(
    pickle_path: str | Path,
    station_ids: list[str],
) -> list[StationSeries]:
    with Path(pickle_path).open("rb") as handle:
        data = pickle.load(handle)

    frames: dict[str, pd.DataFrame] = {}
    for station_id, df in _iter_station_frames(data):
        if station_id in station_ids:
            frames[station_id] = df

    if not frames:
        raise ValueError("No station data found for the provided station ids")

    common_index = None
    for df in frames.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)

    if common_index is None or common_index.empty:
        raise ValueError("No common dates across stations")

    series: list[StationSeries] = []
    for station_id in station_ids:
        df = frames.get(station_id)
        if df is None:
            raise ValueError(f"Missing station {station_id} in pickle")
        df = df.loc[common_index]
        missing_cols = [col for col in DYNAMIC_COLUMNS + STATIC_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Station {station_id} missing columns: {missing_cols}")

        # Check NaNs before dropping
        cols_to_check = DYNAMIC_COLUMNS + [TARGET_COLUMN]
        nan_counts = df[cols_to_check].isna().sum()
        print(f"\nStation {station_id} NaN counts:", nan_counts.sum())
        if nan_counts.sum() > 0:
            print(f"\n=== Station {station_id} ===")
            print("NaN counts per column:")
            print(nan_counts[nan_counts > 0])
            # Print exact locations
            mask = df[cols_to_check].isna()
            for row_idx, row in mask.iterrows():
                missing_in_row = row[row].index.tolist()
                if missing_in_row:
                    print(
                        f"Index={row_idx}, missing columns={missing_in_row}"
                    )

        df = df.dropna(subset=DYNAMIC_COLUMNS + [TARGET_COLUMN])
        dynamic = df[DYNAMIC_COLUMNS].to_numpy(dtype=np.float32)
        target = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
        static = df[STATIC_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
        series.append(
            StationSeries(
                station_id=station_id,
                dynamic=dynamic,
                target=target,
                static=static,
                dates=df.index.to_numpy(),
            )
        )

    return series


def build_short_model_path(model_dir: str | Path, run_name: str) -> Path:
    """Return a short, deterministic checkpoint path safe for Windows path limits."""
    digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:16]
    return Path(model_dir) / f"gnn_{digest}.pt"


def save_temporal_gnn_run(
    model: nn.Module,
    config: TrainConfig,
    model_path: str | Path,
) -> tuple[Path, Path]:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path.with_suffix(".json")
    metadata = config.to_metadata()
    if config.run_name is not None:
        metadata["run_name"] = config.run_name
    try:
        torch.save(model.state_dict(), model_path)
    except RuntimeError as exc:
        resolved = str(model_path.resolve())
        raise RuntimeError(
            f"Failed to save model checkpoint to {resolved} "
            f"(path length={len(resolved)}). "
            "Check that the directory exists, is writable, and the path is not too long."
        ) from exc
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_temporal_gnn_run(
    model_path: str | Path,
    config: TrainConfig,
    *,
    dynamic_dim: int,
    static_dim: int,
    input_dim: int,
    adj_tensor: torch.Tensor | None,
    adj_tensors: list[torch.Tensor] | None,
    device: torch.device,
) -> nn.Module:
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    resolved_config = config
    if metadata_path.exists():
        resolved_config = config_from_saved_metadata(config, model_path)
    model = build_temporal_model(
        resolved_config,
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        input_dim=input_dim,
        adj_tensor=adj_tensor,
        adj_tensors=adj_tensors,
        device=device,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_temporal_gnn_run(
    model: nn.Module,
    *,
    config: TrainConfig,
    series: list[StationSeries],
    station_ids: list[str],
    split_idx: int,
    test_loader: DataLoader,
    weighted_adj: pd.DataFrame,
) -> RunResult:
    test_start_idx = split_idx + config.window_days - 1
    prediction_frames = build_prediction_frames(model, series, config.window_days)
    errors_by_metric = {
        metric: compute_node_errors(model, test_loader, station_ids, error_metric=metric)
        for metric in ERROR_METRICS
    }
    test_start_date = (
        None if config.examine_train_test_peaks else pd.Timestamp(series[0].dates[test_start_idx])
    )
    return_period_values, _ = collect_return_period_nrmse_values(
        prediction_frames,
        station_ids,
        test_start_date=test_start_date,
        examine_train_test=config.examine_train_test_peaks,
        report_missing_points=False,
    )
    return RunResult(
        config=config,
        model=model,
        station_ids=station_ids,
        series=series,
        split_idx=split_idx,
        test_start_idx=test_start_idx,
        test_loader=test_loader,
        weighted_adj=weighted_adj,
        prediction_frames=prediction_frames,
        errors_by_metric=errors_by_metric,
        return_period_values=return_period_values,
    )


def _build_weighted_adj_matrices(config: TrainConfig) -> tuple[pd.DataFrame, torch.Tensor | None, list[torch.Tensor] | None]:
    static_info = config.static_info_path
    if config.several_gnn_layers:
        weighted_adjs = [
            adj_fn(
                station_ids=DEFAULT_STATION_IDS,
                relations=DEFAULT_RELATIONS,
                static_info=static_info,
            )
            for adj_fn in ALL_WEIGHTED_ADJ_FNS
        ]
        weighted_adj = create_weighted_adj_matrix_hydrological(
            station_ids=DEFAULT_STATION_IDS,
            relations=DEFAULT_RELATIONS,
            static_info=static_info,
        )
        adj_tensors = [
            torch.tensor(adj_df.to_numpy(dtype=np.float32)) for adj_df in weighted_adjs
        ]
        return weighted_adj, None, adj_tensors

    weighted_adj = config.resolve_weighted_adj_fn()(
        station_ids=DEFAULT_STATION_IDS,
        relations=DEFAULT_RELATIONS,
        static_info=static_info,
    )
    adj_tensor = torch.tensor(weighted_adj.to_numpy(dtype=np.float32))
    return weighted_adj, adj_tensor, None


def prepare_temporal_gnn_data(config: TrainConfig) -> tuple[
    list[StationSeries],
    list[str],
    int,
    DataLoader,
    DataLoader,
    pd.DataFrame,
    torch.Tensor | None,
    list[torch.Tensor] | None,
    int,
    int,
    int,
]:
    weighted_adj, adj_tensor, adj_tensors = _build_weighted_adj_matrices(config)
    station_ids = list(weighted_adj.index)
    series = load_station_series(config.pickle_path, station_ids)
    total_days = series[0].dynamic.shape[0]
    num_samples = total_days - config.window_days + 1
    if num_samples <= 0:
        raise ValueError("Not enough data to create windows")
    split_idx = int(num_samples * (1 - config.test_fraction))

    if config.uses_mono_encoder():
        train_dataset = GraphWindowDatasetLSTMOnly(series, config.window_days, 0, split_idx)
        test_dataset = GraphWindowDatasetLSTMOnly(series, config.window_days, split_idx, num_samples)
        dynamic_dim = train_dataset.dynamic_dim
        static_dim = train_dataset.static_dim
        input_dim = train_dataset.input_dim
    else:
        train_dataset = GraphWindowDataset(series, config.window_days, 0, split_idx)
        test_dataset = GraphWindowDataset(series, config.window_days, split_idx, num_samples)
        dynamic_dim = train_dataset.dynamic_dim
        static_dim = train_dataset.static_dim
        input_dim = dynamic_dim + static_dim

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return (
        series,
        station_ids,
        split_idx,
        train_loader,
        test_loader,
        weighted_adj,
        adj_tensor,
        adj_tensors,
        dynamic_dim,
        static_dim,
        input_dim,
    )


def _mean_training_metric_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: TrainingMetricLoss,
    device: torch.device,
    *,
    mono: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        if mono:
            for features_batch, y_batch in loader:
                features_batch = features_batch.to(device)
                y_batch = y_batch.to(device)
                preds = _forward_model(model, features_batch)
                total_loss += float(loss_fn(preds, y_batch).item())
                batch_count += 1
        else:
            for dynamic_batch, static_batch, y_batch in loader:
                dynamic_batch = dynamic_batch.to(device)
                static_batch = static_batch.to(device)
                y_batch = y_batch.to(device)
                preds = _forward_model(model, dynamic_batch, static_batch)
                total_loss += float(loss_fn(preds, y_batch).item())
                batch_count += 1
    if batch_count == 0:
        return float("nan")
    return total_loss / batch_count


def _train_one_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    mono: bool,
) -> torch.Tensor:
    if mono:
        features_batch, y_batch = batch
        features_batch = features_batch.to(device)
        y_batch = y_batch.to(device)
        return _forward_model(model, features_batch), y_batch
    dynamic_batch, static_batch, y_batch = batch
    dynamic_batch = dynamic_batch.to(device)
    static_batch = static_batch.to(device)
    y_batch = y_batch.to(device)
    return _forward_model(model, dynamic_batch, static_batch), y_batch


def run_temporal_gnn_training(config: TrainConfig | None = None) -> RunResult:
    config = config or TrainConfig()
    set_random_seed(config.seed)

    (
        series,
        station_ids,
        split_idx,
        train_loader,
        test_loader,
        weighted_adj,
        adj_tensor,
        adj_tensors,
        dynamic_dim,
        static_dim,
        input_dim,
    ) = prepare_temporal_gnn_data(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mono = config.uses_mono_encoder()
    model = build_temporal_model(
        config,
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        input_dim=input_dim,
        adj_tensor=adj_tensor,
        adj_tensors=adj_tensors,
        device=device,
    )

    training_metric = _validate_training_metric(config.training_metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = TrainingMetricLoss(training_metric).to(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds, y_batch = _train_one_batch(model, batch, device, mono=mono)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

        if config.verbose >= 2:
            train_loss = _mean_training_metric_loss(
                model, train_loader, loss_fn, device, mono=mono
            )
            test_loss = _mean_training_metric_loss(
                model, test_loader, loss_fn, device, mono=mono
            )
            message = (
                f"Epoch {epoch:03d}/{config.epochs} "
                f"train_{training_metric}={train_loss:.4f} "
                f"test_{training_metric}={test_loss:.4f}"
            )
            if config.verbose >= 3:
                errors_by_metric = {
                    metric: compute_node_errors(
                        model, test_loader, station_ids, error_metric=metric
                    )
                    for metric in ERROR_METRICS
                }
                metric_parts: list[str] = []
                for metric in ERROR_METRICS:
                    values = [
                        float(value)
                        for value in errors_by_metric[metric].values()
                        if value is not None and np.isfinite(value)
                    ]
                    mean_value = float(np.mean(values)) if values else float("nan")
                    metric_parts.append(f"mean_{metric}={mean_value:.4f}")
                message = f"{message} " + ", ".join(metric_parts)
            print(message)
        elif config.verbose >= 1:
            print(f"Epoch {epoch:03d}/{config.epochs} complete")

    return evaluate_temporal_gnn_run(
        model,
        config=config,
        series=series,
        station_ids=station_ids,
        split_idx=split_idx,
        test_loader=test_loader,
        weighted_adj=weighted_adj,
    )


def config_from_saved_metadata(
    config: TrainConfig,
    model_path: str | Path,
) -> TrainConfig:
    metadata_path = Path(model_path).with_suffix(".json")
    if not metadata_path.exists():
        return config
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    updates: dict[str, Any] = {}
    if metadata.get("weighted_adj") == "multi":
        updates["several_gnn_layers"] = True
    elif "weighted_adj" in metadata:
        updates["weighted_adj_fn"] = _resolve_weighted_adj_fn(metadata["weighted_adj"])
    if "several_gnn_layers" in metadata:
        updates["several_gnn_layers"] = bool(metadata["several_gnn_layers"])
    if "model_type" in metadata:
        updates["model_type"] = _validate_model_type(str(metadata["model_type"]))
    if "training_metric" in metadata:
        updates["training_metric"] = _resolve_saved_training_metric(str(metadata["training_metric"]))
    if "aggregation" in metadata:
        updates["aggregation"] = _validate_aggregation(str(metadata["aggregation"]))
    if "adj_normalization" in metadata:
        updates["adj_normalization"] = _validate_adj_normalization(str(metadata["adj_normalization"]))
    elif "normalize_adj" in metadata:
        normalize_name = str(metadata["normalize_adj"])
        if normalize_name == "_normalize_adj_row_norm":
            updates["adj_normalization"] = "row_norm"
        elif normalize_name == "_normalize_adj_inv_dist":
            updates["adj_normalization"] = "inv_dist"
    if "undirected_graph" in metadata:
        updates["undirected_graph"] = bool(metadata["undirected_graph"])
    if "self_loops" in metadata:
        updates["self_loops"] = bool(metadata["self_loops"])
    if "train_message_passing" in metadata:
        updates["train_message_passing"] = bool(metadata["train_message_passing"])
    if "hidden_dim" in metadata:
        updates["hidden_dim"] = int(metadata["hidden_dim"])
    if "message_passes" in metadata:
        updates["message_passes"] = int(metadata["message_passes"])
    if "epochs" in metadata:
        updates["epochs"] = int(metadata["epochs"])
    if "lr" in metadata:
        updates["lr"] = float(metadata["lr"])
    if "batch_size" in metadata:
        updates["batch_size"] = int(metadata["batch_size"])
    if "window_days" in metadata:
        updates["window_days"] = int(metadata["window_days"])
    if "verbose" in metadata:
        updates["verbose"] = int(metadata["verbose"])
    return replace(config, **updates)


def load_and_evaluate_temporal_gnn_run(
    config: TrainConfig,
    model_path: str | Path,
) -> RunResult:
    config = config_from_saved_metadata(config, model_path)
    (
        series,
        station_ids,
        split_idx,
        _train_loader,
        test_loader,
        weighted_adj,
        adj_tensor,
        adj_tensors,
        dynamic_dim,
        static_dim,
        input_dim,
    ) = prepare_temporal_gnn_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_temporal_gnn_run(
        model_path,
        config,
        dynamic_dim=dynamic_dim,
        static_dim=static_dim,
        input_dim=input_dim,
        adj_tensor=adj_tensor,
        adj_tensors=adj_tensors,
        device=device,
    )
    return evaluate_temporal_gnn_run(
        model,
        config=config,
        series=series,
        station_ids=station_ids,
        split_idx=split_idx,
        test_loader=test_loader,
        weighted_adj=weighted_adj,
    )


def main() -> None:
    config = TrainConfig()
    result = run_temporal_gnn_training(config)
    model = result.model
    series = result.series
    station_ids = result.station_ids
    split_idx = result.split_idx
    test_start_idx = result.test_start_idx
    test_loader = result.test_loader
    weighted_adj = result.weighted_adj
    prediction_frames = result.prediction_frames
    window_days = config.window_days

    if isinstance(model, TemporalGNN):
        print_temporal_gnn_summary(model, window_days=window_days)
    elif isinstance(model, TemporalGNNLSTMOnly):
        print_temporal_gnn_lstm_only_summary(model, window_days=window_days)
    else:
        print(f"Trained model: {model.__class__.__name__}")

    visuals_dir = Path(config.visuals_dir) if config.visuals_dir is not None else None
    if visuals_dir is not None:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    test_years = sorted(set(pd.to_datetime(series[0].dates[test_start_idx:]).year))
    test_start_date = pd.Timestamp(series[0].dates[test_start_idx])
    test_end_date = pd.Timestamp(series[0].dates[-1])
    station_names = load_station_name_map(config.static_info_path)
    plot_test_years_predictions(
        prediction_frames,
        test_years=test_years,
        station_ids=station_ids,
        station_names=station_names,
        output_dir=visuals_dir / "gnn_lstm_test_years" if visuals_dir is not None else None,
        filename_prefix="gnn_lstm",
        year_range=(2016, 2022),
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )

    peak_test_start = (
        None if config.examine_train_test_peaks else pd.Timestamp(series[0].dates[test_start_idx])
    )
    plot_return_period_nrmse_boxplots(
        prediction_frames,
        station_ids=station_ids,
        gnn_model_label="GNN-LSTM",
        test_start_date=peak_test_start,
        examine_train_test=config.examine_train_test_peaks,
        output_path=(
            visuals_dir / "gnn_lstm_return_period_nrmse_boxplots.png"
            if visuals_dir is not None
            else None
        ),
        show_plot=False,
    )
    plot_return_period_nrmse_lineplots(
        prediction_frames,
        station_ids=station_ids,
        gnn_model_label="GNN-LSTM",
        test_start_date=peak_test_start,
        examine_train_test=config.examine_train_test_peaks,
        station_names=station_names,
        output_path=(
            visuals_dir / "gnn_lstm_return_period_nrmse_lineplots.png"
            if visuals_dir is not None
            else None
        ),
        show_plot=False,
    )

    error_metric = "NSE"
    error_by_station = result.errors_by_metric[error_metric]
    output_html = (
        visuals_dir / f"gnn_lstm_{error_metric.lower()}_NSE_map.html"
        if visuals_dir is not None
        else None
    )
    plot_graph_error_map(
        weighted_adj,
        config.static_info_path,
        error_by_station,
        error_metric=error_metric,
        output_html=output_html,
        show_plot=False,
        show_edge_km=False,
        output_png=visuals_dir / f"gnn_lstm_{error_metric.lower()}_NSE_map.png" if visuals_dir is not None else None,
    )

    error_metric = "KGE"
    error_by_station = result.errors_by_metric[error_metric]
    output_html = (
        visuals_dir / f"gnn_lstm_{error_metric.lower()}_KGE_map.html"
        if visuals_dir is not None
        else None
    )
    plot_graph_error_map(
        weighted_adj,
        config.static_info_path,
        error_by_station,
        error_metric=error_metric,
        output_html=output_html,
        show_plot=False,
        show_edge_km=False,
        output_png=visuals_dir / f"gnn_lstm_{error_metric.lower()}_KGE_map.png" if visuals_dir is not None else None,
    )

    if visuals_dir is not None:
        plot_KGE_separated_map(
            weighted_adj,
            config.static_info_path,
            station_frames=prediction_frames,
            output_html=visuals_dir / "gnn_lstm_kge_separated_map.html",
            output_png=visuals_dir / "gnn_lstm_kge_separated_map.png",
            show_edge_km=False,
            show_plot=False,
        )

    nse_by_station = result.errors_by_metric["NSE"]
    conchi_output_html = (
        visuals_dir / "gnn_lstm_nse_conchi_lstm_comparison.html" if visuals_dir is not None else None
    )
    comparison_NSE_conchi(
        weighted_adj,
        config.static_info_path,
        nse_by_station=nse_by_station,
        conchi_model="LSTM",
        conchi_scenario="TS2",
        output_html=conchi_output_html,
        output_png=visuals_dir / "gnn_lstm_nse_conchi_lstm_comparison.png" if visuals_dir is not None else None,
        show_plot=False,
        show_edge_km=False,
    )


def build_graph_sequences(series: list[StationSeries], end_idx: int, window_days: int) -> np.ndarray:
    num_nodes = len(series)
    dynamic_dim = series[0].dynamic.shape[1]
    dynamic = np.zeros((num_nodes, window_days, dynamic_dim), dtype=np.float32)
    start = end_idx - window_days + 1
    for node_idx, station in enumerate(series):
        dynamic[node_idx] = station.dynamic[start : end_idx + 1]
    return dynamic


def build_graph_sequences_with_static(
    series: list[StationSeries],
    end_idx: int,
    window_days: int,
) -> np.ndarray:
    num_nodes = len(series)
    dynamic_dim = series[0].dynamic.shape[1]
    static_dim = series[0].static.shape[0]
    features = np.zeros((num_nodes, window_days, dynamic_dim + static_dim), dtype=np.float32)
    start = end_idx - window_days + 1
    for node_idx, station in enumerate(series):
        features[node_idx] = _broadcast_static_into_window(
            station.dynamic[start : end_idx + 1],
            station.static,
        )
    return features


def _forward_model(
    model: nn.Module,
    features_batch: torch.Tensor,
    static_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_mono_model(model):
        return model(features_batch)
    if static_batch is None:
        raise ValueError("static_batch is required for dual-encoder TemporalGNN models")
    return model(features_batch, static_batch)


def build_prediction_frames(
    model: nn.Module,
    series: list[StationSeries],
    window_days: int,
) -> dict[str, pd.DataFrame]:
    if not series:
        return {}

    station_ids = [s.station_id for s in series]
    dates = series[0].dates
    valid_end = np.arange(window_days - 1, len(dates))
    static_features = np.stack([s.static for s in series]).astype(np.float32)
    device = next(model.parameters()).device
    data = {station_id: {"date": [], "observed": [], "predicted": []} for station_id in station_ids}
    mono = _is_mono_model(model)

    model.eval()
    with torch.no_grad():
        for end_idx in valid_end:
            if mono:
                feature_seq = build_graph_sequences_with_static(series, end_idx, window_days)
                feature_batch = torch.from_numpy(feature_seq[None, :, :, :]).to(device)
                raw_preds = _forward_model(model, feature_batch)
            else:
                dynamic_seq = build_graph_sequences(series, end_idx, window_days)
                dynamic_batch = torch.from_numpy(dynamic_seq[None, :, :, :]).to(device)
                static_batch = torch.from_numpy(static_features[None, :, :]).to(device)
                raw_preds = _forward_model(model, dynamic_batch, static_batch)
            preds = _clamp_streamflow_predictions(raw_preds).cpu().numpy()[0]
            for node_idx, station_id in enumerate(station_ids):
                data[station_id]["date"].append(dates[end_idx])
                data[station_id]["observed"].append(series[node_idx].target[end_idx])
                data[station_id]["predicted"].append(preds[node_idx])

    return {station_id: pd.DataFrame(values) for station_id, values in data.items()}


def compute_node_errors(
    model: nn.Module,
    loader: DataLoader,
    station_ids: list[str],
    error_metric: str = "NSE",
) -> dict[str, float]:
    device = next(model.parameters()).device
    observed_by_station: dict[str, list[float]] = {station_id: [] for station_id in station_ids}
    predicted_by_station: dict[str, list[float]] = {station_id: [] for station_id in station_ids}
    mono = _is_mono_model(model)

    model.eval()
    with torch.no_grad():
        if mono:
            for features_batch, y_batch in loader:
                features_batch = features_batch.to(device)
                y_batch = y_batch.to(device)
                preds = _clamp_streamflow_predictions(_forward_model(model, features_batch))
                for node_idx, station_id in enumerate(station_ids):
                    observed_by_station[station_id].extend(y_batch[:, node_idx].cpu().numpy().tolist())
                    predicted_by_station[station_id].extend(preds[:, node_idx].cpu().numpy().tolist())
        else:
            for dynamic_batch, static_batch, y_batch in loader:
                dynamic_batch = dynamic_batch.to(device)
                static_batch = static_batch.to(device)
                y_batch = y_batch.to(device)
                preds = _clamp_streamflow_predictions(
                    _forward_model(model, dynamic_batch, static_batch)
                )
                for node_idx, station_id in enumerate(station_ids):
                    observed_by_station[station_id].extend(y_batch[:, node_idx].cpu().numpy().tolist())
                    predicted_by_station[station_id].extend(preds[:, node_idx].cpu().numpy().tolist())

    if not any(observed_by_station[station_id] for station_id in station_ids):
        return {station_id: float("nan") for station_id in station_ids}

    return {
        station_id: compute_station_error(
            observed_by_station[station_id],
            predicted_by_station[station_id],
            error_metric,
        )
        for station_id in station_ids
    }


if __name__ == "__main__":
    main()
