#!/usr/bin/env python3
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from adj_matrix_visualize_maps_GNNs import (
    DEFAULT_RELATIONS,
    DEFAULT_STATIC_INFO_PATH,
    create_adj_matrix,
    create_weighted_adj_matrix,
)
from visuals import (
    compute_station_error,
    plot_graph_error_map,
    plot_random_year_predictions,
)


DEFAULT_PICKLE_PATH = "/mnt/d/streamflow_prediction/inputs_allstations_plus_static.pkl"
DEFAULT_MODEL_DIR = "/mnt/d/streamflow_prediction/models/MLP"
DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals"
WINDOW_DAYS = 365
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
TEST_FRACTION = 0.2
SEED = 42

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


@dataclass
class WindowedSample:
    features: np.ndarray
    target: float
    end_date: pd.Timestamp
    station_id: str


def _iter_station_frames(data: dict) -> Iterable[tuple[str, pd.DataFrame]]:
    for station_id, df in data.items():
        if isinstance(df, pd.DataFrame):
            yield station_id, df


def build_samples(df: pd.DataFrame, station_id: str, window_days: int) -> list[WindowedSample]:
    df = df.sort_index()
    df = df.dropna(subset=[TARGET_COLUMN] + DYNAMIC_COLUMNS)

    static_values = df[STATIC_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
    dynamic_values = df[DYNAMIC_COLUMNS].to_numpy(dtype=np.float32)
    target_values = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    dates = df.index.to_numpy()

    samples: list[WindowedSample] = []
    for start in range(0, len(df) - window_days + 1):
        end = start + window_days
        window = dynamic_values[start:end].reshape(-1)
        features = np.concatenate([window, static_values], axis=0)
        target = float(target_values[end - 1])
        samples.append(
            WindowedSample(
                features=features,
                target=target,
                end_date=pd.Timestamp(dates[end - 1]),
                station_id=station_id,
            )
        )
    return samples


def train_test_split_time(samples: list[WindowedSample], test_fraction: float) -> tuple[list[WindowedSample], list[WindowedSample]]:
    samples_sorted = sorted(samples, key=lambda s: s.end_date)
    split_idx = int(len(samples_sorted) * (1 - test_fraction))
    return samples_sorted[:split_idx], samples_sorted[split_idx:]


def to_loader(samples: list[WindowedSample], batch_size: int, shuffle: bool) -> DataLoader:
    x = np.stack([s.features for s in samples]).astype(np.float32)
    y = np.array([s.target for s in samples], dtype=np.float32).reshape(-1, 1)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_model(model: nn.Module, model_dir: str | Path | None) -> Path | None:
    if model_dir is None:
        return None
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mlp_streamflow.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(model_path: str | Path, input_dim: int, device: torch.device) -> MLP:
    model = MLP(input_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _group_samples_by_station(samples: list[WindowedSample]) -> dict[str, list[WindowedSample]]:
    grouped: dict[str, list[WindowedSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.station_id, []).append(sample)
    return grouped


def build_prediction_frames(
    model: MLP,
    samples: list[WindowedSample],
) -> dict[str, pd.DataFrame]:
    if not samples:
        return {}

    grouped = _group_samples_by_station(samples)
    device = next(model.parameters()).device
    data_by_station: dict[str, pd.DataFrame] = {}

    model.eval()
    with torch.no_grad():
        for station_id, station_samples in grouped.items():
            station_samples.sort(key=lambda s: s.end_date)
            x = np.stack([s.features for s in station_samples]).astype(np.float32)
            preds = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            data_by_station[station_id] = pd.DataFrame(
                {
                    "date": [s.end_date for s in station_samples],
                    "observed": [s.target for s in station_samples],
                    "predicted": preds,
                }
            )

    return data_by_station


def compute_errors_by_station(
    model: MLP,
    samples: list[WindowedSample],
    error_metric: str = "NSE",
) -> dict[str, float]:
    if not samples:
        return {}

    grouped = _group_samples_by_station(samples)
    device = next(model.parameters()).device
    errors_by_station: dict[str, float] = {}

    model.eval()
    with torch.no_grad():
        for station_id, station_samples in grouped.items():
            if not station_samples:
                continue
            x = np.stack([s.features for s in station_samples]).astype(np.float32)
            y = np.array([s.target for s in station_samples], dtype=np.float32)
            preds = model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
            errors_by_station[station_id] = (
                compute_station_error(y, preds, error_metric) if y.size else float("nan")
            )

    return errors_by_station


def main() -> None:
    pickle_path = Path(DEFAULT_PICKLE_PATH)
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)


    all_samples: list[WindowedSample] = []
    for station_id, df in _iter_station_frames(data):
        if not all(col in df.columns for col in DYNAMIC_COLUMNS + STATIC_COLUMNS + [TARGET_COLUMN]):
            continue
        samples = build_samples(df, station_id, WINDOW_DAYS)
        all_samples.extend(samples)
        print(f"{station_id}: {len(samples)} samples")

    train_samples, test_samples = train_test_split_time(all_samples, TEST_FRACTION)
    print(f"Total samples: {len(all_samples)} | Train: {len(train_samples)} | Test: {len(test_samples)}")

    train_loader = to_loader(train_samples, BATCH_SIZE, shuffle=True)
    test_loader = to_loader(test_samples, BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_samples[0].features.shape[0]
    model = MLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                loss = loss_fn(preds, y_batch)
                test_loss_sum += loss.item() * x_batch.size(0)
        test_loss = test_loss_sum / len(test_loader.dataset)

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

    model_path = save_model(model, DEFAULT_MODEL_DIR)
    if model_path is not None:
        print(f"Saved model to {model_path}")

    visuals_dir = Path(DEFAULT_VISUALS_DIR) if DEFAULT_VISUALS_DIR is not None else None
    if visuals_dir is not None:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = build_prediction_frames(model, all_samples)
    plot_random_year_predictions(
        prediction_frames,
        visuals_dir=visuals_dir,
        seed=SEED,
        filename_prefix="mlp",
    )

    adj_matrix = create_adj_matrix(
        station_ids=None,
        relations=DEFAULT_RELATIONS,
        static_info=DEFAULT_STATIC_INFO_PATH,
    )
    weighted_adj = create_weighted_adj_matrix(adj_matrix, DEFAULT_STATIC_INFO_PATH)
    error_metric = "NSE"
    error_by_station = compute_errors_by_station(model, test_samples, error_metric=error_metric)
    error_by_station = {
        station_id: error_by_station.get(station_id, float("nan"))
        for station_id in weighted_adj.index
    }
    output_html = (
        visuals_dir / f"mlp_{error_metric.lower()}_map.html" if visuals_dir is not None else None
    )
    plot_graph_error_map(
        weighted_adj,
        DEFAULT_STATIC_INFO_PATH,
        error_by_station,
        error_metric=error_metric,
        output_html=output_html,
        show_plot=False,
    )


if __name__ == "__main__":
    main()
