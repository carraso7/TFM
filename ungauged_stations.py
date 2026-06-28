#!/usr/bin/env python3
"""Train a temporal GNN on a reduced graph and evaluate on the full network.

Simulates an ungauged-station scenario by excluding one station from the
training adjacency while keeping downstream routing intact, then evaluates
the trained model on the complete six-node graph including the held-out station.
"""
from __future__ import annotations

from pathlib import Path

from adj_matrix_visualize_maps_GNNs import (
    DEFAULT_RELATIONS,
    plot_weighted_graph_map,
)
from GNN_predictor_temporal_lstm import (
    DEFAULT_PICKLE_PATH,
    DEFAULT_STATIC_INFO_PATH,
    RunResult,
    TrainConfig,
    build_weighted_adj_for_relations,
    evaluate_temporal_gnn_run,
    plot_temporal_gnn_visuals,
    prepare_temporal_gnn_eval_data,
    run_temporal_gnn_training,
    set_model_adjacency,
)

# Exclude station 018 from the training graph (271 linked directly to 170):
# UNGAUGED_TRAIN_RELATIONS: list[tuple[str, str]] = [
#     ("061", "170"),
#     ("080", "062"),
#     ("062", "170"),
#     ("271", "170"),
# ]
# UNGAUGED_STATION_ID = "018"  # Jaca

# Exclude station 062 from the training graph (080 linked directly to 170):
UNGAUGED_TRAIN_RELATIONS: list[tuple[str, str]] = [
    ("061", "170"),
    ("018", "170"),
    ("271", "018"),
    ("080", "170"),
]
UNGAUGED_STATION_ID = "062"  # Binies

DEFAULT_VISUALS_DIR = "/mnt/d/streamflow_prediction/visuals/ungauged_outs"


def plot_ungauged_training_graph(
    train_result: RunResult,
    config: TrainConfig,
    *,
    visuals_dir: str | Path,
) -> None:
    """Export an interactive map of the training graph used for ungauged runs.

    Args:
        train_result: Training run containing the weighted adjacency matrix.
        config: Training configuration with paths to static station metadata.
        visuals_dir: Directory where HTML and QGIS GeoPackage outputs are written.
    """
    visuals_path = Path(visuals_dir)
    visuals_path.mkdir(parents=True, exist_ok=True)
    plot_weighted_graph_map(
        train_result.weighted_adj,
        config.static_info_path,
        output_html=visuals_path / "ungauged_training_graph.html",
        output_qgis_dir=visuals_path / "for_QGIS" / "ungauged_training_graph",
        qgis_gpkg_name="ungauged_training_graph.gpkg",
        show_plot=False,
    )


def evaluate_ungauged_on_full_graph(train_result: RunResult) -> RunResult:
    """Evaluate a model trained on a reduced graph against the full station set.

    Reloads evaluation data with the default six-station relations, swaps the
    model adjacency to the full graph, and prints per-station NSE/KGE metrics.

    Args:
        train_result: Output of ``run_temporal_gnn_training`` on the reduced graph.

    Returns:
        Evaluation ``RunResult`` with errors for all stations in the full graph.

    Raises:
        NotImplementedError: If the trained model uses multiple adjacency layers.
        ValueError: If a single adjacency tensor cannot be built for evaluation.
    """
    config = train_result.config
    if config.several_gnn_layers:
        raise NotImplementedError(
            "Ungauged full-graph evaluation does not support multi-adjacency models yet"
        )

    eval_series, eval_station_ids, eval_test_loader, eval_weighted_adj = (
        prepare_temporal_gnn_eval_data(
            config,
            DEFAULT_RELATIONS,
            train_result.split_idx,
        )
    )
    _, eval_adj_tensor, _ = build_weighted_adj_for_relations(config, DEFAULT_RELATIONS)
    if eval_adj_tensor is None:
        raise ValueError("Expected a single adjacency tensor for full-graph evaluation")

    set_model_adjacency(train_result.model, eval_adj_tensor, config)
    eval_result = evaluate_temporal_gnn_run(
        train_result.model,
        config=config,
        series=eval_series,
        station_ids=eval_station_ids,
        split_idx=train_result.split_idx,
        test_loader=eval_test_loader,
        weighted_adj=eval_weighted_adj,
    )

    print(
        f"\nFull-graph evaluation metrics "
        f"(including previously ungauged station {UNGAUGED_STATION_ID}):"
    )
    for metric in ("NSE", "KGE"):
        print(f"  {metric}:")
        for station_id in eval_station_ids:
            value = eval_result.errors_by_metric[metric].get(station_id)
            marker = "  <-- ungauged during training" if station_id == UNGAUGED_STATION_ID else ""
            if value is None or not isinstance(value, float):
                print(f"    {station_id}: N/A{marker}")
            else:
                print(f"    {station_id}: {value:.4f}{marker}")

    return eval_result


def main() -> None:
    """Train on the ungauged graph, evaluate on the full graph, and save visuals."""
    train_config = TrainConfig(
        pickle_path=DEFAULT_PICKLE_PATH,
        static_info_path=DEFAULT_STATIC_INFO_PATH,
        relations=UNGAUGED_TRAIN_RELATIONS,
        visuals_dir=DEFAULT_VISUALS_DIR,
    )
    visuals_path = Path(DEFAULT_VISUALS_DIR)
    visuals_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on 5-node graph (station {UNGAUGED_STATION_ID} excluded; "
        f"080 linked to 170; 271→018→170 preserved)..."
    )
    train_result = run_temporal_gnn_training(train_config)
    print(
        f"Training graph stations ({len(train_result.station_ids)}): "
        f"{', '.join(train_result.station_ids)}"
    )

    print(
        f"\nEvaluating on full 6-node graph (station {UNGAUGED_STATION_ID} included)..."
    )
    eval_result = evaluate_ungauged_on_full_graph(train_result)

    plot_ungauged_training_graph(train_result, train_config, visuals_dir=visuals_path)

    plot_temporal_gnn_visuals(
        eval_result,
        train_config,
        visuals_dir=visuals_path,
        filename_prefix="ungauged",
        gnn_model_label="GNN-LSTM (ungauged train)",
        year_range=(None, None),
        show_model_summary=False,
    )


if __name__ == "__main__":
    main()
