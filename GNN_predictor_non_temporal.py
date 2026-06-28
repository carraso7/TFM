#!/usr/bin/env python3
"""Non-temporal GNN for multi-station daily streamflow prediction.

Builds graph-window datasets from pickle inputs, trains a message-passing GNN
with MLP encoder/decoder, and exports prediction and error visualizations.
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from visuals import (
    comparison_NSE_conchi,
    compute_station_error,
    load_station_name_map,
    plot_graph_error_map,
    plot_random_year_predictions,
    plot_return_period_nrmse_boxplots,
    plot_return_period_nrmse_lineplots,
    plot_test_years_predictions,
)

from adj_matrix_visualize_maps_GNNs import (
    DEFAULT_RELATIONS,
    DEFAULT_STATIC_INFO_PATH,
    DEFAULT_STATION_IDS,
    create_adj_matrix,
    create_weighted_adj_matrix,
)

### TODO REVISAR ZURIZA, SE PRECICE MUCHO MEJOR CON EL ARCHIVO NO CHECKED QUE CON EL CHECKED.
### TODO PONER EL RESTO DE LOS 3 CODIGOS CON LOS DATOS NUEVOS SELECCIONADOS. 

DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_allstations_plus_static.pkl" # This one has wrong data. Take data from inputs_allstations_plus_static_checked.pkl, (it does not have jaca on it)
# DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_selected_stations.pkl"
DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals"
WINDOW_DAYS = 365
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
TEST_FRACTION = 0.2
SEED = 42
HIDDEN_DIM = 64
MESSAGE_PASSES = 3

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


torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


@dataclass
class StationSeries:
    """
    Container for preprocessed time-series data of a single hydrological station.

    Attributes:
        station_id (str): Unique identifier for the gauging station (e.g. "E001").
        dynamic (np.ndarray): Shape (T, D) float32 array of daily meteorological
            features, where T is the number of days and D = len(DYNAMIC_COLUMNS).
            Column order matches DYNAMIC_COLUMNS: pr, tmax_total, tmin_total,
            Humidity, SPEI, nao, WEMO.
        target (np.ndarray): Shape (T,) float32 array of daily streamflow values
            (m³/s or equivalent units stored in the source pickle).
        static (np.ndarray): Shape (S,) float32 array of time-invariant catchment
            descriptors. Column order matches STATIC_COLUMNS: Catchment Area,
            Elevation, Agricultural areas, Forests, Shrub/herbaceous vegetation.
        dates (np.ndarray): Shape (T,) array of numpy datetime64 dates aligned
            with `dynamic` and `target`.
    """
    station_id: str
    dynamic: np.ndarray
    target: np.ndarray
    static: np.ndarray
    dates: np.ndarray


def _iter_station_frames(data: dict) -> Iterable[tuple[str, pd.DataFrame]]:
    """
    Yield only the DataFrame entries from the raw pickle dictionary.

    The pickle may contain non-DataFrame values (metadata, scalars, etc.).
    This helper filters them out so downstream code only sees tabular data.

    Args:
        data (dict): Raw dictionary loaded from the pickle file. Keys are
            station IDs (str); values may be pd.DataFrame or other types.

    Yields:
        tuple[str, pd.DataFrame]: (station_id, dataframe) pairs where the
            value is a pd.DataFrame.
    """
    for station_id, df in data.items():
        if isinstance(df, pd.DataFrame):
            yield station_id, df


def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Apply symmetric self-loop addition followed by row-normalization to an
    adjacency matrix.

    Steps performed in order:
        1. Add self-loops by setting the diagonal to 1.0 (in-place on a clone).
        2. Compute the row-sum degree vector D.
        3. Divide each row by its degree so every row sums to 1.0.
           Rows with a zero sum (isolated nodes) are left unchanged (divided by 1).

    This is the standard "left" normalization used in GCN-style message passing:
        Â = D⁻¹ · (A + I)

    Args:
        adj (torch.Tensor): Square adjacency matrix of shape (N, N), dtype float.
            May be weighted or binary; need not be symmetric.

    Returns:
        torch.Tensor: Row-normalized adjacency matrix of shape (N, N), same dtype
            and device as the input.
    """

    ### Pensar si normalizar adj:
    """
    Add self-loops and apply row-normalization to an adjacency matrix.

    Steps performed in order:
        1. Clone the input to avoid modifying the original tensor.
        2. Add self-loops by setting the diagonal to 1.0, so each node
           always aggregates its own embedding during message passing.
        3. Compute the row-sum degree vector D.
        4. Divide each row by its degree.
           Isolated nodes (row-sum == 0) are left unchanged.

    The resulting formula is:
        Â = D⁻¹ · (A + I)

    ⚠️  Behaviour with distance-weighted edges
    -------------------------------------------
    When `adj` contains distance-based weights (e.g. produced by
    `create_weighted_adj_matrix`), normalization preserves the *relative*
    influence of each neighbour within a row but removes the *absolute*
    magnitude of those weights.

    Concretely, two nodes with different numbers of neighbours will both
    produce row sums of 1.0 after normalization, even if their raw weight
    distributions differ substantially. This is intentional for training
    stability — it prevents nodes with many connections from dominating
    the hidden state — but it means the aggregated embedding is a
    *weighted average* of neighbour embeddings, not a *weighted sum*.

    If the raw edge weights carry meaningful absolute strength (e.g. a very
    close upstream station should exert strong influence regardless of the
    total number of connections), consider skipping normalization or using
    symmetric normalization (D⁻¹/² · A · D⁻¹/²) instead.

    Args:
        adj (torch.Tensor): Square adjacency matrix of shape (N, N), dtype float.
            May be binary or real-valued (e.g. inverse-distance weights).
            Need not be symmetric.

    Returns:
        torch.Tensor: Row-normalized adjacency of shape (N, N), same dtype and
            device as the input. Every non-isolated row sums to 1.0.
    """

    adj = adj.clone()
    adj.fill_diagonal_(1.0)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


def _clamp_streamflow_predictions(preds: torch.Tensor) -> torch.Tensor:
    """Inference-only floor at zero; training still uses unconstrained model outputs."""
    return torch.clamp(preds, min=0.0)


def _normalize_adj_only_diag(adj: torch.Tensor) -> torch.Tensor:
    """
    Apply symmetric self-loop addition followed by row-normalization to an
    adjacency matrix.

    Steps performed in order:
        1. Add self-loops by setting the diagonal to 1.0 (in-place on a clone).
        2. Compute the row-sum degree vector D.
        3. Divide each row by its degree so every row sums to 1.0.
           Rows with a zero sum (isolated nodes) are left unchanged (divided by 1).

    This is the standard "left" normalization used in GCN-style message passing:
        Â = D⁻¹ · (A + I)

    Args:
        adj (torch.Tensor): Square adjacency matrix of shape (N, N), dtype float.
            May be weighted or binary; need not be symmetric.

    Returns:
        torch.Tensor: Row-normalized adjacency matrix of shape (N, N), same dtype
            and device as the input.
    """

    ### Pensar si normalizar adj:
    """
    Add self-loops and apply row-normalization to an adjacency matrix.

    Steps performed in order:
        1. Clone the input to avoid modifying the original tensor.
        2. Add self-loops by setting the diagonal to 1.0, so each node
           always aggregates its own embedding during message passing.
        3. Compute the row-sum degree vector D.
        4. Divide each row by its degree.
           Isolated nodes (row-sum == 0) are left unchanged.

    The resulting formula is:
        Â = D⁻¹ · (A + I)

    ⚠️  Behaviour with distance-weighted edges
    -------------------------------------------
    When `adj` contains distance-based weights (e.g. produced by
    `create_weighted_adj_matrix`), normalization preserves the *relative*
    influence of each neighbour within a row but removes the *absolute*
    magnitude of those weights.

    Concretely, two nodes with different numbers of neighbours will both
    produce row sums of 1.0 after normalization, even if their raw weight
    distributions differ substantially. This is intentional for training
    stability — it prevents nodes with many connections from dominating
    the hidden state — but it means the aggregated embedding is a
    *weighted average* of neighbour embeddings, not a *weighted sum*.

    If the raw edge weights carry meaningful absolute strength (e.g. a very
    close upstream station should exert strong influence regardless of the
    total number of connections), consider skipping normalization or using
    symmetric normalization (D⁻¹/² · A · D⁻¹/²) instead.

    Args:
        adj (torch.Tensor): Square adjacency matrix of shape (N, N), dtype float.
            May be binary or real-valued (e.g. inverse-distance weights).
            Need not be symmetric.

    Returns:
        torch.Tensor: Row-normalized adjacency of shape (N, N), same dtype and
            device as the input. Every non-isolated row sums to 1.0.
    """

    adj = adj.clone()
    adj.fill_diagonal_(1.0)
    # row_sum = adj.sum(dim=1, keepdim=True)
    # row_sum[row_sum == 0] = 1.0
    return adj # / row_sum


def _normalize_adj_row_norm(adj: torch.Tensor, self_loops: bool = True) -> torch.Tensor:
    """Row-normalize an adjacency matrix, optionally adding self-loops first.

    Args:
        adj: Square adjacency tensor of shape ``(N, N)``.
        self_loops: If ``True``, set the diagonal to 1.0 before normalization.

    Returns:
        Row-normalized adjacency with the same shape and dtype as ``adj``.
    """
    adj = adj.clone()
    if self_loops:
        adj.fill_diagonal_(1.0)
    row_sum = adj.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


def _normalize_adj_inv_dist(adj: torch.Tensor, self_loops: bool = True) -> torch.Tensor:
    """Add self-loops if self_loops is True and convert non-zero edge weights to inverse distance (1/value)."""
    adj = adj.clone()
    if self_loops:
        adj.fill_diagonal_(1.0)
    nonzero = adj != 0
    adj[nonzero] = 1.0 / adj[nonzero]
    return adj

DEFAULT_NORMALIZE_ADJ = _normalize_adj_inv_dist

class GraphWindowDataset(Dataset):
    """
    PyTorch Dataset that slides a fixed-length window over synchronized
    multi-station time series and returns graph-level feature/target pairs.

    Each sample corresponds to one calendar day `t` in [start_idx, end_idx).
    The feature for node i is the flattened dynamic window of the preceding
    `window_days` days concatenated with the station's static descriptor:
        x_i = [dynamic[t-W+1 : t+1].reshape(-1), static_i]   shape: (W*D + S,)

    The target for node i is the streamflow value on day t:
        y_i = target[t]   scalar float32

    Args:
        series (list[StationSeries]): List of N pre-loaded station objects,
            all sharing the same time axis (same length T).
        window_days (int): Length W of the look-back window in days.
        start_idx (int): First valid *end* index (0-based) within the time axis.
            Samples are drawn from start_idx to end_idx - 1.
        end_idx (int): One-past-the-last valid end index.

    Attributes:
        feature_dim (int): Total feature size per node = window_days * D + S,
            where D = number of dynamic features and S = number of static features.
    """
    def __init__(
        self,
        series: list[StationSeries],
        window_days: int,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Initialize a graph-window dataset split over synchronized stations."""
        self.series = series
        self.window_days = window_days
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.num_nodes = len(series)
        self.feature_dim = window_days * series[0].dynamic.shape[1] + series[0].static.shape[0]

    def __len__(self) -> int:
        """Return the number of samples in this split (end_idx - start_idx)."""
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the feature matrix and target vector for sample `idx`. Returns the 
        target streamflows and the time series of the previous `window_days` days 
        for each station, concatenated with the static features. 

        Args:
            idx (int): Zero-based sample index within [0, len(self)). 

        Returns:
            tuple:
                - features (torch.Tensor): Shape (N, feature_dim), float32.
                  Row i contains the flattened window + static features for
                  station i.
                - targets (torch.Tensor): Shape (N,), float32. Entry i is the
                  streamflow of station i on the last day of the window.
        """
        end = self.start_idx + idx + self.window_days
        start = end - self.window_days
        features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        targets = np.zeros((self.num_nodes,), dtype=np.float32)
        for node_idx, station in enumerate(self.series):
            window = station.dynamic[start:end].reshape(-1)
            features[node_idx] = np.concatenate([window, station.static], axis=0)
            targets[node_idx] = station.target[end - 1]
        return torch.from_numpy(features), torch.from_numpy(targets)


class MLPEncoder(nn.Module):
    """
    Two-layer MLP that projects per-node raw features into a hidden embedding.

    Architecture:
        Linear(input_dim → hidden_dim) → ReLU →
        Linear(hidden_dim → hidden_dim) → ReLU

    Args:
        input_dim (int): Size of the input feature vector per node
            (= window_days * D + S).
        hidden_dim (int): Size of the output embedding per node.

    Forward:
        x (torch.Tensor): Shape (..., input_dim).

    Returns:
        torch.Tensor: Shape (..., hidden_dim). Node embeddings before
            message passing.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map node features to hidden embeddings."""
        return self.net(x)


class MLPDecoder(nn.Module):
    """
    Two-layer MLP that maps a node embedding to a scalar streamflow prediction.

    Architecture:
        Linear(hidden_dim → hidden_dim) → ReLU →
        Linear(hidden_dim → 1)

    Args:
        hidden_dim (int): Dimensionality of the input embedding.

    Forward:
        x (torch.Tensor): Shape (..., hidden_dim).

    Returns:
        torch.Tensor: Shape (..., 1). Raw (un-activated) streamflow predictions.
    """
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Linear layers only operate on the last dimension, so we can feed in tensors of shape (..., hidden_dim) and it will apply the same transformation to every node in every batch.
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map node embeddings to scalar streamflow predictions."""
        return self.net(x)


class SimpleGNN(nn.Module):
    """
    Graph Neural Network for multi-station streamflow prediction.

    The model encodes per-node feature vectors, performs `message_passes`
    rounds of neighbourhood aggregation over a fixed adjacency matrix,
    then decodes each node embedding to a scalar prediction.

    Architecture:
        1. MLPEncoder   : (N, input_dim)  →  (N, hidden_dim)
        2. Message pass : repeat `message_passes` times
               h ← ReLU(Â · h)      [batched einsum over B samples]
        3. MLPDecoder   : (N, hidden_dim) →  (N,)

    The adjacency matrix is stored as a non-trainable buffer (row-normalized
    via `_normalize_adj`) so it moves to the correct device with .to(device).

    Args:
        input_dim (int): Feature dimension per node.
        hidden_dim (int): Hidden embedding dimension used throughout.
        message_passes (int): Number of graph propagation steps (graph depth).
        adj (torch.Tensor): Shape (N, N) float adjacency matrix (weighted or
            binary). Will be row-normalized and registered as a buffer.

    Forward:
        x (torch.Tensor): Shape (B, N, input_dim), float32. Batch of B
            graph snapshots, each with N node feature vectors.

    Returns:
        torch.Tensor: Shape (B, N), float32. Predicted streamflow for every
            node in every sample of the batch.
    """
    def __init__(self, input_dim: int, hidden_dim: int, message_passes: int, adj: torch.Tensor) -> None:
        """Register encoder/decoder modules and the normalized adjacency buffer."""
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim)
        self.decoder = MLPDecoder(hidden_dim)
        self.message_passes = message_passes
        # print(adj) ### TODO PRINT QUITAR
        # print(_normalize_adj(adj))
        self.register_buffer("adj", DEFAULT_NORMALIZE_ADJ(adj)) ### TODO Pensar si normalizar adj
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode nodes, propagate messages, and decode streamflow predictions."""
        h = self.encoder(x)
        for _ in range(self.message_passes):
            h = h + torch.relu(torch.einsum("ij,bjh->bih", self.adj, h))
        return self.decoder(h).squeeze(-1)


def load_station_series(
    pickle_path: str | Path,
    station_ids: list[str],
) -> list[StationSeries]:
    """
    Load, align, and package station time-series data from a pickle file.

    The function:
        1. Loads a dict {station_id: pd.DataFrame} from `pickle_path`.
        2. Filters to the requested `station_ids`.
        3. Computes the intersection of all date indices to ensure alignment.
        4. Extracts dynamic features, static features, and the target column
           for each station, dropping rows with NaN in dynamic or target columns.

    Expected pickle structure:
        dict[str, pd.DataFrame] where each DataFrame is indexed by date
        (pd.DatetimeIndex or compatible) and contains at minimum the columns
        listed in DYNAMIC_COLUMNS, STATIC_COLUMNS, and TARGET_COLUMN.

    Args:
        pickle_path (str | Path): Path to the pickle file containing the
            station data dictionary.
        station_ids (list[str]): Ordered list of station identifiers to load.
            The order is preserved in the returned list.

    Returns:
        list[StationSeries]: One StationSeries per station, in the same order
            as `station_ids`. All series share the same aligned time axis.

    Raises:
        ValueError: If no station data is found, if the common date index is
            empty, if a requested station is missing, or if required columns
            are absent from any station's DataFrame.
    """
    with Path(pickle_path).open("rb") as handle:
        data = pickle.load(handle)

    frames: dict[str, pd.DataFrame] = {}
    for station_id, df in _iter_station_frames(data):
        if station_id in station_ids:
            frames[station_id] = df

    if not frames:
        raise ValueError("No station data found for the provided station ids")

    # Variable to store the common index of the dates of all stations, it will be a list of datas common to all stations.
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

def print_model_summary(model: SimpleGNN, batch_size: int = 1) -> None:
    """
    Print a structured summary of the SimpleGNN model.

    Displays layer names, output shapes (inferred from a dummy forward pass),
    parameter counts, and high-level hyperparameters stored on the model.

    Args:
        model (SimpleGNN): The GNN model to summarise.
        batch_size (int): Batch size used for the dummy forward pass (default 1).
    """
    num_nodes   = model.adj.shape[0]
    input_dim   = model.encoder.net[0].in_features
    hidden_dim  = model.encoder.net[0].out_features

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    sep = "=" * 62
    print(sep)
    print(f"  SimpleGNN — Model Summary")
    print(sep)
    print(f"  Nodes (stations)   : {num_nodes}")
    print(f"  Input dim / node   : {input_dim}")
    print(f"  Hidden dim         : {hidden_dim}")
    print(f"  Message-pass steps : {model.message_passes}")
    print(sep)
    print(f"  {'Component':<28} {'Output shape':<20} {'Params':>8}")
    print("-" * 62)

    device = next(model.parameters()).device
    dummy  = torch.zeros(batch_size, num_nodes, input_dim, device=device)

    # Encoder
    with torch.no_grad():
        h = model.encoder(dummy)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"  {'MLPEncoder':<28} {str(tuple(h.shape)):<20} {enc_params:>8,}")

    # Message-passing (no extra params — uses adj buffer)
    for i in range(model.message_passes):
        h = torch.relu(torch.einsum("ij,bjh->bih", model.adj, h))
        print(f"  {'  GCN pass ' + str(i+1):<28} {str(tuple(h.shape)):<20} {'—':>8}")

    # Decoder
    out = model.decoder(h)
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  {'MLPDecoder':<28} {str(tuple(out.squeeze(-1).shape)):<20} {dec_params:>8,}")

    print(sep)
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Trainable params   : {trainable_params:,}")
    print(f"  Non-trainable      : {total_params - trainable_params:,}  (adj matrix buffer)")
    print(sep)

def main() -> None:
    """Train the default non-temporal GNN and generate evaluation plots."""
    adj_matrix = create_adj_matrix(
        station_ids=DEFAULT_STATION_IDS,
        relations=DEFAULT_RELATIONS,
        static_info=DEFAULT_STATIC_INFO_PATH,
    )
    weighted_adj = create_weighted_adj_matrix(adj_matrix, DEFAULT_STATIC_INFO_PATH)
    station_ids = list(weighted_adj.index)

    series = load_station_series(DEFAULT_PICKLE_PATH, station_ids)
    total_days = series[0].dynamic.shape[0]
    print("total days in dataset: ", total_days) ### TODO PRINT QUITAR
    num_samples = total_days - WINDOW_DAYS + 1
    if num_samples <= 0:
        raise ValueError("Not enough data to create windows")

    split_idx = int(num_samples * (1 - TEST_FRACTION))
    train_dataset = GraphWindowDataset(series, WINDOW_DAYS, 0, split_idx)
    test_dataset = GraphWindowDataset(series, WINDOW_DAYS, split_idx, num_samples)
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_tensor = torch.tensor(weighted_adj.to_numpy(dtype=np.float32))
    input_dim = train_dataset.feature_dim
    model = SimpleGNN(input_dim, HIDDEN_DIM, MESSAGE_PASSES, adj_tensor).to(device)
    print_model_summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train() # Set the model to training mode at the start of each epoch
        train_loss_sum = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad() # Clear gradients before the backward pass to not accumulate them across batches
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward() # Compute gradients with respect to the loss
            optimizer.step() # Update model parameters based on the computed gradients
            train_loss_sum += loss.item() * x_batch.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval() # Set the model to evaluation mode before running inference on the test set
        test_loss_sum = 0.0
        with torch.no_grad(): # Disable gradient computation for inference to save memory and computations since we are not updating the model during evaluation
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)
                test_loss_sum += loss.item() * x_batch.size(0)
        test_loss = test_loss_sum / len(test_loader.dataset)

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

    visuals_dir = Path(DEFAULT_VISUALS_DIR) if DEFAULT_VISUALS_DIR is not None else None
    if visuals_dir is not None:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = build_prediction_frames(model, series, WINDOW_DAYS)
    # plot_random_year_predictions(
    #     prediction_frames,
    #     visuals_dir=visuals_dir,
    #     seed=SEED,
    #     filename_prefix="gnn",
    # )

    # print(prediction_frames) ### TODO PRINT QUITAR
    test_start_idx = split_idx + WINDOW_DAYS - 1
    test_years = sorted(set(pd.to_datetime(series[0].dates[test_start_idx:]).year))
    station_names = load_station_name_map(DEFAULT_STATIC_INFO_PATH)
    plot_test_years_predictions(
        prediction_frames,
        test_years=test_years,
        station_ids=station_ids,
        station_names=station_names,
        output_dir=visuals_dir / "gnn_test_years" if visuals_dir is not None else None,
        filename_prefix="non_temp_gnn",
        year_range=(2016, 2022),
    )

    plot_return_period_nrmse_boxplots(
        prediction_frames,
        station_ids=station_ids,
        gnn_model_label="GNN-LSTM",
        test_start_date=pd.Timestamp(series[0].dates[test_start_idx]),
        output_path=(
            visuals_dir / "gnn_return_period_nrmse_boxplots.png"
            if visuals_dir is not None
            else None
        ),
        show_plot=False,
    )
    plot_return_period_nrmse_lineplots(
        prediction_frames,
        station_ids=station_ids,
        gnn_model_label="GNN-LSTM",
        test_start_date=pd.Timestamp(series[0].dates[test_start_idx]),
        station_names=station_names,
        output_path=(
            visuals_dir / "gnn_return_period_nrmse_lineplots.png"
            if visuals_dir is not None
            else None
        ),
        show_plot=False,
    )

    error_metric = "NSE"
    error_metric = "nRMSE"
    error_metric = "KGE"
    error_by_station = compute_node_errors(model, test_loader, station_ids, error_metric=error_metric)
    output_html = (
        visuals_dir / f"gnn_{error_metric.lower()}_map.html" if visuals_dir is not None else None
    )
    plot_graph_error_map(
        weighted_adj,
        DEFAULT_STATIC_INFO_PATH,
        error_by_station,
        error_metric=error_metric,
        output_html=output_html,
        show_plot=False,
    )

    nse_by_station = compute_node_errors(model, test_loader, station_ids, error_metric="NSE")
    conchi_output_html = (
        visuals_dir / "gnn_nse_conchi_MC_LSTM_comparison.html" if visuals_dir is not None else None
    )
    comparison_NSE_conchi(
        weighted_adj,
        DEFAULT_STATIC_INFO_PATH,
        nse_by_station=nse_by_station,
        conchi_model="MC-LSTM",
        conchi_scenario="TS2",
        output_html=conchi_output_html,
        show_plot=False,
    )


def build_graph_features(series: list[StationSeries], end_idx: int, window_days: int) -> np.ndarray:
    """
    Build a single-snapshot node feature matrix for inference or evaluation.

    For each station node, concatenates the flattened dynamic window ending at
    `end_idx` with the station's static descriptor:
        x_i = [dynamic[end_idx - window_days + 1 : end_idx + 1].reshape(-1),
               static_i]

    Args:
        series (list[StationSeries]): List of N station objects sharing a
            common time axis.
        end_idx (int): Index of the last day (inclusive, 0-based) of the
            look-back window.
        window_days (int): Length W of the look-back window in days.

    Returns:
        np.ndarray: Shape (N, W*D + S), float32. Row i is the feature vector
            for station i, where D = number of dynamic features and
            S = number of static features.
    """
    num_nodes = len(series)
    feature_dim = window_days * series[0].dynamic.shape[1] + series[0].static.shape[0]
    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    start = end_idx - window_days + 1
    for node_idx, station in enumerate(series):
        window = station.dynamic[start : end_idx + 1].reshape(-1)
        features[node_idx] = np.concatenate([window, station.static], axis=0)
    return features


def build_prediction_frames(
    model: SimpleGNN,
    series: list[StationSeries],
    window_days: int,
) -> dict[str, pd.DataFrame]:
    """
    Run the trained model over all valid time steps and collect predictions.

    Iterates through every valid end index (from window_days-1 to T-1),
    builds the graph feature snapshot, feeds it through the model, and
    records observed vs. predicted streamflow for every station.

    Args:
        model (SimpleGNN): Trained GNN model in eval mode (set inside this
            function). Must already reside on the correct device.
        series (list[StationSeries]): Station data used to build features.
            All stations must share the same time axis.
        window_days (int): Look-back window length W used during training.

    Returns:
        dict[str, pd.DataFrame]: One DataFrame per station keyed by station_id.
            Each DataFrame has three columns:
                - "date"      : numpy datetime64 values.
                - "observed"  : float32 measured streamflow.
                - "predicted" : float32 model output (un-denormalized).
            Returns an empty dict if `series` is empty.
    """
    if not series:
        return {}

    station_ids = [s.station_id for s in series]
    dates = series[0].dates
    valid_end = np.arange(window_days - 1, len(dates))
    device = next(model.parameters()).device
    data = {station_id: {"date": [], "observed": [], "predicted": []} for station_id in station_ids}

    model.eval()
    with torch.no_grad():
        for end_idx in valid_end:
            features = build_graph_features(series, end_idx, window_days)
            raw_preds = model(torch.from_numpy(features).unsqueeze(0).to(device))
            # Inference-only clamp: plots and exported predictions never go below zero.
            preds = _clamp_streamflow_predictions(raw_preds).cpu().numpy()[0]
            for node_idx, station_id in enumerate(station_ids):
                data[station_id]["date"].append(dates[end_idx])
                data[station_id]["observed"].append(series[node_idx].target[end_idx])
                data[station_id]["predicted"].append(preds[node_idx])

    return {station_id: pd.DataFrame(values) for station_id, values in data.items()}


def compute_node_errors(
    model: SimpleGNN,
    loader: DataLoader,
    station_ids: list[str],
    error_metric: str = "NSE",
) -> dict[str, float]:
    """
    Compute per-node error metrics on a given DataLoader split.

    Args:
        model (SimpleGNN): Trained GNN model. Placed in eval mode internally.
        loader (DataLoader): DataLoader yielding (features, targets) batches
            where targets have shape (B, N).
        station_ids (list[str]): Ordered list of N station identifiers. Must
            match the node order used when building the dataset.
        error_metric (str): One of "MSE", "NSE", "RMSE", "nRMSE", or "KGE".

    Returns:
        dict[str, float]: Maps each station_id to its scalar error value.
            Returns NaN for all stations if the loader is empty.
    """
    device = next(model.parameters()).device
    observed_by_station: dict[str, list[float]] = {station_id: [] for station_id in station_ids}
    predicted_by_station: dict[str, list[float]] = {station_id: [] for station_id in station_ids}

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            raw_preds = model(x_batch)
            # Inference-only clamp: NSE maps and error metrics use non-negative predictions.
            preds = _clamp_streamflow_predictions(raw_preds)
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
