#!/usr/bin/env python3
from __future__ import annotations

"""Iterative one-at-a-time hyperparameter search for Temporal GNN-LSTM models.

Orchestrates training runs, evaluation, visualization export, and baseline
updates across multiple sweep iterations. Each iteration varies one parameter
at a time from a fixed baseline, writes per-run diagnostics under a visuals
root directory, and optionally promotes improved settings to the next
iteration's baseline.

Output directory layout (under ``visuals_root``)::

    iter_<n>/
        baseline/<run_name>/          # baseline run plots
        <param_name>/<run_name>/      # one folder per varied parameter value
        summaries/
            param_search_summary.csv  # all runs in the iteration
            best_params.csv             # best value per swept parameter
            baseline_update.csv         # baseline promotion decision
            *_boxplot_by_value.png      # cross-value comparison plots
            all_params_*_boxplot_grid.png

CSV files use semicolon (``;``) separators. ``param_search_summary.csv`` rows
contain parameter values, ``varied_param`` / ``varied_value``, per-metric
``mean_*`` and ``var_*`` columns (station-aggregated), return-period peak
``mean_peak_*_nRMSE`` / ``var_peak_*_nRMSE`` columns, and ``run_name``.

Model checkpoints are stored under ``models_dir/iter_<n>/`` as ``.pt`` files
named via :func:`build_run_name`.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adj_matrix_visualize_maps_GNNs import (
    create_weighted_adj_matrix_all_paths,
    create_weighted_adj_matrix_all_river_distances,
    create_weighted_adj_matrix_dense,
    create_weighted_adj_matrix_hydrological,
)
from GNN_predictor_temporal_lstm import (
    VALID_TRAINING_METRICS,
    WEIGHTED_ADJ_ALIAS_BY_NAME,
    RunResult,
    TrainConfig,
    build_short_model_path,
    load_and_evaluate_temporal_gnn_run,
    run_temporal_gnn_training,
    save_temporal_gnn_run,
)
from visuals import (
    DEFAULT_RETURN_PERIODS,
    ERROR_METRICS,
    FONTSIZE_DEFAULT,
    _format_return_period_label,
    comparison_NSE_conchi,
    compute_station_metric_boxplot_ylim,
    load_station_name_map,
    plot_graph_error_map,
    plot_KGE_separated_map,
    plot_return_period_nrmse_boxplots,
    plot_return_period_nrmse_lineplots,
    plot_station_metric_boxplot_by_param_values,
    plot_station_metric_boxplot_grid,
    plot_test_years_predictions,
)

BASELINE_NAME = "GNN_lstm"
DEFAULT_MODELS_DIR = "/mnt/d/streamflow_prediction/models"
DEFAULT_VISUALS_ROOT = "/mnt/d/streamflow_prediction/visuals/GNN_lstm_params_test"
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_COMPARISON_METRIC = "NSE"
DEFAULT_HIGHLIGHT_BASELINE_IN_BOXPLOTS = True

UNIFIED_BOXPLOT_YMIN = -0.5


@dataclass
class ParamOption:
    """A search-space option pairing a raw value with a display label.

    Attributes:
        value: Parameter value passed to training (e.g. a callable adjacency
            builder, int, float, bool, or str).
        label: Human-readable label used in CSV output and run naming; when
            ``None``, a label is derived from ``value``.
    """

    value: Any
    label: str | None = None


@dataclass
class BaselineUpdateDecision:
    """Result of evaluating whether to advance the sweep baseline.

    Attributes:
        should_continue: Whether another iteration should run.
        strategy: How the baseline changed: ``"combined_best"``,
            ``"single_param"``, or ``"stop"``.
        baseline: Updated (or unchanged) parameter dictionary for the next
            iteration.
        baseline_result: Evaluation artifacts for the chosen baseline run.
        baseline_model_path: Filesystem path to the baseline checkpoint.
        baseline_nse: Mean station NSE of the new baseline.
        previous_baseline_nse: Mean station NSE before the update attempt.
        selected_param: Parameter name updated when ``strategy`` is
            ``"single_param"``; otherwise ``None``.
        selected_value_label: Display label of the promoted value for
            ``"single_param"`` updates; otherwise ``None``.
    """

    should_continue: bool
    strategy: str
    baseline: dict[str, Any]
    baseline_result: RunResult
    baseline_model_path: Path
    baseline_nse: float
    previous_baseline_nse: float
    selected_param: str | None = None
    selected_value_label: str | None = None


BASELINE: dict[str, Any] = {
    "epochs": 60, #30,
    "hidden_dim": 64,
    "lr": 1e-3,
    "batch_size": 16, #32,
    "message_passes": 3,
    "window_days": 1460, #365,
    "weighted_adj": ParamOption(create_weighted_adj_matrix_all_river_distances, "river_dist"),
    "aggregation": "max", #"sum",
    "train_error_metric": "RMSE",
    "model_type": ParamOption("dual", "dual"),
    "several_gnn_layers": False,
    "train_message_passing": True,
    "undirected_graph": False,
    "adj_normalization": "inv_dist",
}

SEARCH_PARAMS = [
    "epochs",
    "hidden_dim",
    "lr",
    "batch_size",
    "message_passes",
    "window_days",
    "weighted_adj",
    "aggregation",
    "train_error_metric",
    "several_gnn_layers",
    "train_message_passing",
    "undirected_graph",
    "adj_normalization",
]

SEARCH_SPACE: dict[str, list[Any]] = {
    "epochs": [30, 60, 120],
    "hidden_dim": [32, 64, 128, 256],
    "lr": [1e-4, 1e-3, 1e-2],
    "batch_size": [16, 32, 64],
    "message_passes": [1, 3, 6, 12],
    "window_days": [180, 365, 730, 1460],
    "weighted_adj": [
        ParamOption(create_weighted_adj_matrix_all_river_distances, "river_dist"),
        ParamOption(create_weighted_adj_matrix_all_paths, "all_paths"),
        ParamOption(create_weighted_adj_matrix_hydrological, "hydro"),
        ParamOption(create_weighted_adj_matrix_dense, "dense"),
    ],
    "aggregation": ["sum", "max", "mean"],
    "train_error_metric": list(VALID_TRAINING_METRICS),
    "model_type": [
        ParamOption("dual", "dual"),
        ParamOption("mono", "mono"),
    ],
    "several_gnn_layers": [False, True],
    "train_message_passing": [True, False],
    "undirected_graph": [False, True],
    "adj_normalization": ["inv_dist", "row_norm"],
}


def resolve_param_option(option: Any) -> tuple[Any, str]:
    """Unwrap a search option into its training value and display label.

    Args:
        option: A :class:`ParamOption` or a bare parameter value.

    Returns:
        Tuple of ``(value, label)`` where ``value`` is the object used for
        training and ``label`` is the string shown in summaries and plots.
    """
    if isinstance(option, ParamOption):
        label = option.label or param_display_label(option.value)
        return option.value, label
    return option, param_display_label(option)


def format_param_value(value: Any) -> str:
    """Format a scalar parameter value for display.

    Args:
        value: Numeric or other scalar to stringify.

    Returns:
        Floats use scientific notation when very small or large; other types
        use ``str(value)``.
    """
    if isinstance(value, float):
        return f"{value:.0e}" if value < 0.01 or value >= 1 else f"{value:g}"
    return str(value)


def param_display_label(value: Any) -> str:
    """Build a consistent display label for any parameter value.

    Args:
        value: Parameter value, :class:`ParamOption`, bool, callable adjacency
            function, or scalar.

    Returns:
        Label string suitable for CSV columns and plot legends.
    """
    if isinstance(value, ParamOption):
        return value.label or param_display_label(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if callable(value):
        fn_name = getattr(value, "__name__", "")
        return WEIGHTED_ADJ_ALIAS_BY_NAME.get(fn_name, fn_name or format_param_value(value))
    return format_param_value(value)


def _abbreviate_word(word: str) -> str:
    """Abbreviate a single token for compact run names.

    Args:
        word: Underscore-separated word fragment.

    Returns:
        First character uppercased plus remaining characters lowercased, or
        empty string for blank input.
    """
    word = word.strip()
    if not word:
        return ""
    if len(word) == 1:
        return word.upper()
    return word[0].upper() + word[1].lower()


def abbreviate_identifier(name: str) -> str:
    """Build a compact token from an underscore-separated identifier.

    Args:
        name: Identifier such as ``"hidden_dim"`` or ``"river_dist"``.

    Returns:
        Concatenation of abbreviated words (e.g. ``"HiDi"`` for
        ``"hidden_dim"``).
    """
    parts = name.replace("-", "_").split("_")
    return "".join(_abbreviate_word(part) for part in parts if part)


def abbreviate_param_value(value: Any) -> str:
    """Abbreviate a parameter value for use in run name segments.

    Args:
        value: Parameter value or :class:`ParamOption`.

    Returns:
        Short token: ``"T"``/``"F"`` for bools, numeric strings unchanged,
        otherwise an abbreviated identifier derived from the display label.
    """
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, ParamOption):
        return abbreviate_param_value(value.label or value.value)
    if callable(value):
        fn_name = getattr(value, "__name__", "")
        alias = WEIGHTED_ADJ_ALIAS_BY_NAME.get(fn_name, fn_name)
        return abbreviate_identifier(alias)
    label = param_display_label(value)
    if label in {"true", "false"}:
        return "T" if label == "true" else "F"
    numeric_candidate = label.replace(".", "").replace("e", "").replace("-", "").replace("+", "")
    if numeric_candidate.isdigit():
        return label
    return abbreviate_identifier(label)


def abbreviate_param_segment(param_name: str, value: Any) -> str:
    """Combine abbreviated parameter name and value into one run-name segment.

    Args:
        param_name: Hyperparameter key (e.g. ``"lr"``).
        value: Parameter value to abbreviate.

    Returns:
        String like ``"Lr1e-3"`` or ``"Hi64"``.
    """
    return f"{abbreviate_identifier(param_name)}{abbreviate_param_value(value)}"


def resolve_bool_param(value: Any) -> bool:
    """Coerce a parameter value to bool, unwrapping :class:`ParamOption`.

    Args:
        value: Boolean or wrapped boolean option.

    Returns:
        ``bool(value)`` after unwrapping.
    """
    if isinstance(value, ParamOption):
        return bool(value.value)
    return bool(value)


def resolve_model_type_param(value: Any) -> str:
    """Resolve model type to a string, unwrapping :class:`ParamOption`.

    Args:
        value: Model type string or wrapped option (e.g. ``"dual"``, ``"mono"``).

    Returns:
        Model type name as ``str``.
    """
    if isinstance(value, ParamOption):
        return str(value.value)
    return str(value)


def params_have_several_gnn_layers(params: dict[str, Any]) -> bool:
    """Check whether the configuration uses multiple GNN layers.

    Args:
        params: Hyperparameter dictionary.

    Returns:
        ``True`` when ``several_gnn_layers`` is enabled.
    """
    return resolve_bool_param(params.get("several_gnn_layers", False))


def serialize_param_for_csv(value: Any) -> Any:
    """Serialize a parameter value for CSV export.

    Args:
        value: Parameter value to record.

    Returns:
        Display label string (via :func:`param_display_label`).
    """
    return param_display_label(value)


def build_param_labels(params: dict[str, Any]) -> dict[str, str]:
    """Map each parameter key to its display label.

    Args:
        params: Hyperparameter dictionary.

    Returns:
        Dictionary with the same keys and string labels as values.
    """
    return {key: param_display_label(value) for key, value in params.items()}


def build_run_name(baseline_name: str, params: dict[str, Any]) -> str:
    """Build a unique, filesystem-safe run identifier from hyperparameters.

    Args:
        baseline_name: Prefix name for the model family (e.g. ``"GNN_lstm"``).
        params: Full hyperparameter dictionary for the run.

    Returns:
        Underscore-separated abbreviated name used for checkpoints and output
        folders. Omits ``weighted_adj`` when ``several_gnn_layers`` is True.
    """
    ordered_keys = [
        "epochs",
        "hidden_dim",
        "lr",
        "batch_size",
        "message_passes",
        "window_days",
        "weighted_adj",
        "aggregation",
        "train_error_metric",
        "model_type",
        "several_gnn_layers",
        "train_message_passing",
        "undirected_graph",
        "adj_normalization",
    ]
    parts = [abbreviate_identifier(baseline_name)]
    for key in ordered_keys:
        if key == "weighted_adj" and params_have_several_gnn_layers(params):
            continue
        if key in params:
            parts.append(abbreviate_param_segment(key, params[key]))
    return "_".join(parts)


def build_train_config(
    params: dict[str, Any],
    *,
    run_name: str,
    models_dir: str | Path,
    examine_train_test_peaks: bool,
) -> TrainConfig:
    """Construct a :class:`TrainConfig` from a hyperparameter dictionary.

    Args:
        params: Hyperparameter dictionary (keys match ``SEARCH_PARAMS``).
        run_name: Identifier passed to the trainer for logging and paths.
        models_dir: Directory where model checkpoints are saved.
        examine_train_test_peaks: Whether return-period peak metrics include
            the training period.

    Returns:
        Config object consumed by :func:`run_temporal_gnn_training` and
        :func:`load_and_evaluate_temporal_gnn_run`.
    """
    several_gnn_layers = params_have_several_gnn_layers(params)
    weighted_adj_value = None
    if not several_gnn_layers:
        weighted_adj_value, _ = resolve_param_option(params["weighted_adj"])
    return TrainConfig(
        epochs=int(params["epochs"]),
        hidden_dim=int(params["hidden_dim"]),
        lr=float(params["lr"]),
        batch_size=int(params["batch_size"]),
        message_passes=int(params["message_passes"]),
        window_days=int(params["window_days"]),
        weighted_adj_fn=weighted_adj_value if callable(weighted_adj_value) else None,
        aggregation=str(params["aggregation"]),
        training_metric=str(params["train_error_metric"]),
        model_type=resolve_model_type_param(params["model_type"]),
        several_gnn_layers=several_gnn_layers,
        train_message_passing=resolve_bool_param(params["train_message_passing"]),
        undirected_graph=resolve_bool_param(params["undirected_graph"]),
        adj_normalization=str(params["adj_normalization"]),
        model_dir=models_dir,
        examine_train_test_peaks=examine_train_test_peaks,
        run_name=run_name,
    )


def _finite_values(values: dict[str, float]) -> list[float]:
    """Extract finite numeric values from a per-station metric dictionary.

    Args:
        values: Mapping of station id to metric value (may contain ``None``
            or non-finite entries).

    Returns:
        List of finite floats; empty when no valid values exist.
    """
    return [float(value) for value in values.values() if value is not None and np.isfinite(value)]


def mean_nse_from_result(result: RunResult) -> float:
    """Compute mean station NSE from a training/evaluation result.

    Args:
        result: Completed run with ``errors_by_metric["NSE"]`` per station.

    Returns:
        Mean NSE across stations with finite values, or ``nan`` if none.
    """
    values = _finite_values(result.errors_by_metric.get("NSE", {}))
    return float(np.mean(values)) if values else float("nan")


def mean_nse_from_row(row: dict[str, Any]) -> float:
    """Read mean NSE from a summary CSV row dictionary.

    Args:
        row: Row dict produced by :func:`summarize_run_row` (expects
            ``"mean_NSE"`` key).

    Returns:
        Finite ``mean_NSE`` value, or ``nan`` if missing or non-finite.
    """
    value = row.get("mean_NSE", float("nan"))
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def param_option_matches(baseline: dict[str, Any], param_name: str, option: Any) -> bool:
    """Check whether a search option equals the baseline value for one param.

    Args:
        baseline: Current baseline hyperparameters.
        param_name: Key to compare.
        option: Candidate value from ``SEARCH_SPACE``.

    Returns:
        ``True`` when display labels match for ``baseline[param_name]`` and
        ``option``.
    """
    return param_display_label(baseline[param_name]) == param_display_label(option)


def baseline_signature(baseline: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Build a hashable signature for comparing parameter sets.

    Args:
        baseline: Hyperparameter dictionary.

    Returns:
        Tuple of ``(param_name, label_or_value)`` pairs for keys in
        ``SEARCH_PARAMS`` present in ``baseline``, ordered consistently.
    """
    signature: list[tuple[str, Any]] = []
    for key in SEARCH_PARAMS:
        if key not in baseline:
            continue
        value, label = resolve_param_option(baseline[key])
        signature.append((key, label if key in SEARCH_SPACE else value))
    return tuple(signature)


def params_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Test whether two hyperparameter dictionaries are equivalent.

    Args:
        left: First parameter set.
        right: Second parameter set.

    Returns:
        ``True`` when :func:`baseline_signature` matches for both.
    """
    return baseline_signature(left) == baseline_signature(right)


def resolve_param_from_label(param_name: str, value_label: str) -> Any:
    """Look up a search-space option by its display label.

    Args:
        param_name: Hyperparameter key in ``SEARCH_SPACE``.
        value_label: Display label to match (as stored in summary CSVs).

    Returns:
        The matching option from ``SEARCH_SPACE[param_name]`` (may be a
        :class:`ParamOption` or bare value).

    Raises:
        ValueError: If no option in the search space has the given label.
    """
    for option in SEARCH_SPACE[param_name]:
        _, label = resolve_param_option(option)
        if label == value_label:
            return option
    raise ValueError(f"No search option found for {param_name}={value_label!r}")


def summarize_run_row(
    *,
    run_name: str,
    varied_param: str,
    varied_value_label: str,
    params: dict[str, Any],
    errors_by_metric: dict[str, dict[str, float]],
    return_period_values: dict[float, dict[str, float]],
) -> dict[str, Any]:
    """Build one summary row for ``param_search_summary.csv``.

    Args:
        run_name: Unique run identifier from :func:`build_run_name`.
        varied_param: Parameter varied in this run, or ``"baseline"``.
        varied_value_label: Display label of the varied value.
        params: Full hyperparameter dictionary for the run.
        errors_by_metric: Per-metric station scores (e.g. ``{"NSE": {id: val}}``).
        return_period_values: Return-period peak nRMSE by period and station;
            keys are return periods (float), values are station-id dicts.

    Returns:
        Flat dict with serialized params, ``varied_param``, ``varied_value``,
        ``mean_*`` / ``var_*`` for each entry in ``ERROR_METRICS``, peak
        nRMSE aggregates for ``DEFAULT_RETURN_PERIODS``, and ``run_name``.
    """
    row: dict[str, Any] = {
        "varied_param": varied_param,
        "varied_value": varied_value_label,
    }
    for key, value in params.items():
        row[key] = serialize_param_for_csv(value)

    for metric in ERROR_METRICS:
        values = _finite_values(errors_by_metric.get(metric, {}))
        row[f"mean_{metric}"] = float(np.mean(values)) if values else float("nan")
        row[f"var_{metric}"] = float(np.var(values)) if values else float("nan")

    for return_period in DEFAULT_RETURN_PERIODS:
        period_label = _format_return_period_label(return_period)
        column_prefix = f"peak_{period_label}_nRMSE"
        values = _finite_values(return_period_values.get(float(return_period), {}))
        row[f"mean_{column_prefix}"] = float(np.mean(values)) if values else float("nan")
        row[f"var_{column_prefix}"] = float(np.var(values)) if values else float("nan")

    row["run_name"] = run_name
    return row


def write_run_visuals(
    result: RunResult,
    *,
    run_name: str,
    visuals_root: Path,
    varied_param: str,
    comparison_metric: str,
    examine_train_test_peaks: bool,
) -> None:
    """Export diagnostic plots and maps for a single training run.

    Writes under ``visuals_root / varied_param / run_name /``:

    * Test-year hydrograph PNGs (``*_test_years/``)
    * Return-period nRMSE boxplots and line plots
    * Graph error map (HTML + PNG) for ``comparison_metric``
    * KGE-separated map and Conchi NSE comparison map

    Args:
        result: Evaluated run containing predictions, adjacency, and metrics.
        run_name: Folder and filename prefix for outputs.
        visuals_root: Iteration-level visuals directory (e.g. ``iter_1``).
        varied_param: Subfolder name grouping runs by swept parameter.
        comparison_metric: Metric used for the graph error map (e.g. ``"NSE"``).
        examine_train_test_peaks: If True, peak metrics span train+test;
            otherwise peaks are test-only.
    """
    run_dir = visuals_root / varied_param / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = result.config
    series = result.series
    station_ids = result.station_ids
    test_start_idx = result.test_start_idx
    prediction_frames = result.prediction_frames
    weighted_adj = result.weighted_adj
    station_names = load_station_name_map(config.static_info_path)

    test_years = sorted(set(pd.to_datetime(series[0].dates[test_start_idx:]).year))
    test_start_date = pd.Timestamp(series[0].dates[test_start_idx])
    test_end_date = pd.Timestamp(series[0].dates[-1])
    plot_test_years_predictions(
        prediction_frames,
        test_years=test_years,
        station_ids=station_ids,
        station_names=station_names,
        output_dir=run_dir / f"{run_name}_test_years",
        filename_prefix=run_name,
        year_range=(2016, 2022),
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        show_plot=False,
    )

    peak_test_start = (
        None
        if examine_train_test_peaks
        else pd.Timestamp(series[0].dates[test_start_idx])
    )
    plot_return_period_nrmse_boxplots(
        prediction_frames,
        station_ids=station_ids,
        gnn_model_label=BASELINE_NAME,
        test_start_date=peak_test_start,
        examine_train_test=examine_train_test_peaks,
        output_path=run_dir / f"{run_name}_return_period_nrmse_boxplots.png",
        show_plot=False,
    )
    plot_return_period_nrmse_lineplots(
        prediction_frames,
        station_ids=station_ids,
        gnn_model_label=BASELINE_NAME,
        test_start_date=peak_test_start,
        examine_train_test=examine_train_test_peaks,
        station_names=station_names,
        output_path=run_dir / f"{run_name}_return_period_nrmse_lineplots.png",
        show_plot=False,
    )

    plot_graph_error_map(
        weighted_adj,
        config.static_info_path,
        result.errors_by_metric.get(comparison_metric, {}),
        error_metric=comparison_metric,
        output_html=run_dir / f"{varied_param}_{comparison_metric}_error_map.html",
        output_png=run_dir / f"{run_name}_{comparison_metric}_error_map.png",
        show_errors=True,
        show_edge_km=True,
        show_plot=False,
    )
    plot_KGE_separated_map(
        weighted_adj,
        config.static_info_path,
        station_frames=prediction_frames,
        output_html=run_dir / f"{varied_param}_KGE_separated_map.html",
        output_png=run_dir / f"{run_name}_KGE_separated_map.png",
        show_edge_km=True,
        show_plot=False,
    )
    comparison_NSE_conchi(
        weighted_adj,
        config.static_info_path,
        nse_by_station=result.errors_by_metric.get("NSE", {}),
        output_html=run_dir / f"{varied_param}_conchi_nse_comparison.html",
        output_png=run_dir / f"{run_name}_conchi_nse_comparison.png",
        show_edge_km=True,
        show_plot=False,
    )


def write_summary_csv(
    run_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write iteration summary rows to a semicolon-separated CSV.

    Args:
        run_rows: List of dicts from :func:`summarize_run_row`.
        output_path: Destination ``.csv`` path (parent dirs are created).

    Returns:
        Resolved output path. Columns match keys in ``run_rows``; ``run_name``
        is placed last when present.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(run_rows)
    if not runs_df.empty and "run_name" in runs_df.columns:
        ordered_columns = [column for column in runs_df.columns if column != "run_name"] + ["run_name"]
        runs_df = runs_df[ordered_columns]
    runs_df.to_csv(output_path, index=False, sep=";")
    return output_path


def write_best_params_csv(
    run_rows: list[dict[str, Any]],
    search_params: list[str],
    baseline: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write per-parameter best values ranked by mean NSE.

    Args:
        run_rows: Iteration summary rows (must include ``varied_param``,
            ``varied_value``, and ``mean_NSE``).
        search_params: Parameter names to evaluate.
        baseline: Current baseline for the ``baseline_value`` column.
        output_path: Destination ``best_params.csv`` path.

    Returns:
        Resolved output path. CSV columns: ``param_name``, ``baseline_value``,
        ``new_value`` (display label of the best run for that parameter).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_rows: list[dict[str, str]] = []
    for param_name in search_params:
        if param_name == "weighted_adj" and params_have_several_gnn_layers(baseline):
            continue
        param_runs = [
            row
            for row in run_rows
            if row.get("varied_param") == param_name and np.isfinite(mean_nse_from_row(row))
        ]
        if not param_runs:
            continue
        best_run = max(param_runs, key=mean_nse_from_row)
        best_rows.append(
            {
                "param_name": param_name,
                "baseline_value": serialize_param_for_csv(baseline[param_name]),
                "new_value": str(best_run["varied_value"]),
            }
        )
    pd.DataFrame(best_rows).to_csv(output_path, index=False, sep=";")
    return output_path


def write_baseline_update_csv(
    decision: BaselineUpdateDecision,
    output_path: str | Path,
) -> Path:
    """Record baseline promotion outcome for one iteration.

    Args:
        decision: Result from :func:`update_baseline_after_iteration`.
        output_path: Destination ``baseline_update.csv`` path.

    Returns:
        Resolved output path. Single-row CSV with strategy, NSE deltas,
        selected parameter (if any), serialized baseline params, and
        ``run_name``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "strategy": decision.strategy,
        "should_continue": decision.should_continue,
        "previous_baseline_nse": decision.previous_baseline_nse,
        "new_baseline_nse": decision.baseline_nse,
        "nse_delta": decision.baseline_nse - decision.previous_baseline_nse,
        "selected_param": decision.selected_param or "",
        "selected_value": decision.selected_value_label or "",
        "run_name": build_run_name(BASELINE_NAME, decision.baseline),
    }
    for key in SEARCH_PARAMS:
        if key in decision.baseline:
            row[key] = serialize_param_for_csv(decision.baseline[key])
    pd.DataFrame([row]).to_csv(output_path, index=False, sep=";")
    return output_path


def train_or_load_run(
    params: dict[str, Any],
    *,
    run_name: str,
    iter_models_dir: Path,
    examine_train_test_peaks: bool,
    skip_existing: bool,
    trained_runs: dict[str, tuple[RunResult, Path]],
    description: str,
) -> tuple[RunResult, Path]:
    """Train a model or load an existing checkpoint for one parameter set.

    Args:
        params: Hyperparameters for the run.
        run_name: Checkpoint basename and cache key.
        iter_models_dir: Iteration-specific models directory.
        examine_train_test_peaks: Passed through to :func:`build_train_config`.
        skip_existing: When True, load from disk if ``.pt`` exists instead of
            retraining.
        trained_runs: In-memory cache updated with ``(result, path)`` on success.
        description: Log prefix describing the run context.

    Returns:
        Tuple of evaluation :class:`RunResult` and checkpoint :class:`Path`.
    """
    if run_name in trained_runs:
        print(f"Reusing cached run {run_name} ({description})")
        return trained_runs[run_name]

    model_path = build_short_model_path(iter_models_dir, run_name)
    legacy_model_path = iter_models_dir / f"{run_name}.pt"
    config = build_train_config(
        params,
        run_name=run_name,
        models_dir=iter_models_dir,
        examine_train_test_peaks=examine_train_test_peaks,
    )

    print(f"\n=== {description}: {run_name} ===")
    print(f"Checkpoint path: {model_path}")
    if skip_existing and model_path.exists():
        print(f"Skipping training; loading existing model from {model_path}")
        result = load_and_evaluate_temporal_gnn_run(config, model_path)
    elif skip_existing and legacy_model_path.exists():
        print(f"Skipping training; loading existing legacy model from {legacy_model_path}")
        result = load_and_evaluate_temporal_gnn_run(config, legacy_model_path)
        model_path = legacy_model_path
    else:
        result = run_temporal_gnn_training(config)
        save_temporal_gnn_run(result.model, config, model_path)

    trained_runs[run_name] = (result, model_path)
    return result, model_path


def build_combined_params_from_best(
    run_rows: list[dict[str, Any]],
    search_params: list[str],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Merge the best per-parameter values into one hyperparameter set.

    For each parameter in ``search_params``, selects the varied value with
    highest ``mean_NSE`` among iteration rows; unchanged keys keep baseline
    values when no valid run exists.

    Args:
        run_rows: Iteration summary rows.
        search_params: Parameters to optimize independently.
        baseline: Starting configuration copied before updates.

    Returns:
        New parameter dictionary with winning options from ``SEARCH_SPACE``.
    """
    combined = dict(baseline)
    for param_name in search_params:
        param_runs = [
            row
            for row in run_rows
            if row.get("varied_param") == param_name and np.isfinite(mean_nse_from_row(row))
        ]
        if not param_runs:
            continue
        best_run = max(param_runs, key=mean_nse_from_row)
        combined[param_name] = resolve_param_from_label(param_name, str(best_run["varied_value"]))
    return combined


def find_best_single_param_improvement(
    run_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    baseline_nse: float,
) -> tuple[str, str, dict[str, Any], float] | None:
    """Find the single-parameter change with the largest NSE gain over baseline.

    Args:
        run_rows: Iteration summary rows (excludes ``baseline`` and
            ``combined_best`` pseudo-runs).
        baseline: Current baseline hyperparameters.
        baseline_nse: Mean NSE of the current baseline.

    Returns:
        ``(param_name, value_label, best_row, new_nse)`` when a strictly
        better single-param run exists; otherwise ``None``.
    """
    best_row: dict[str, Any] | None = None
    best_delta = 0.0

    for row in run_rows:
        varied_param = str(row.get("varied_param", ""))
        if varied_param in {"", "baseline", "combined_best"}:
            continue
        if param_display_label(baseline[varied_param]) == str(row["varied_value"]):
            continue
        row_nse = mean_nse_from_row(row)
        if not np.isfinite(row_nse):
            continue
        delta = row_nse - baseline_nse
        if delta > best_delta:
            best_delta = delta
            best_row = row

    if best_row is None:
        return None
    varied_param = str(best_row["varied_param"])
    return (
        varied_param,
        str(best_row["varied_value"]),
        best_row,
        mean_nse_from_row(best_row),
    )


def update_baseline_after_iteration(
    *,
    baseline: dict[str, Any],
    baseline_result: RunResult,
    baseline_model_path: Path,
    iteration_rows: list[dict[str, Any]],
    search_params: list[str],
    iter_models_dir: Path,
    examine_train_test_peaks: bool,
    skip_existing: bool,
    trained_runs: dict[str, tuple[RunResult, Path]],
    iteration: int,
) -> BaselineUpdateDecision:
    """Decide how to update the baseline after one sweep iteration.

    Tries combined-best params first (train if changed); if mean NSE improves,
    promotes that set. Otherwise tries the best single-parameter improvement.
    Stops the sweep when neither strategy beats the current baseline.

    Args:
        baseline: Hyperparameters at the start of the iteration.
        baseline_result: Evaluation result for the iteration baseline run.
        baseline_model_path: Checkpoint path for the baseline model.
        iteration_rows: Summary rows from :func:`run_one_iteration`.
        search_params: Parameters included in the sweep.
        iter_models_dir: Directory for iteration model checkpoints.
        examine_train_test_peaks: Peak evaluation mode for any new training.
        skip_existing: Whether to reuse existing checkpoints.
        trained_runs: Cache of already-trained runs in this iteration.
        iteration: Current iteration index (for logging).

    Returns:
        :class:`BaselineUpdateDecision` describing the next baseline and
        whether to continue.
    """
    previous_baseline_nse = mean_nse_from_result(baseline_result)
    combined_params = build_combined_params_from_best(iteration_rows, search_params, baseline)
    combined_run_name = build_run_name(BASELINE_NAME, combined_params)

    if params_equal(combined_params, baseline):
        combined_result = baseline_result
        combined_path = baseline_model_path
        combined_nse = previous_baseline_nse
        print(
            f"\nCombined best params match the current baseline "
            f"(mean NSE={combined_nse:.4f})."
        )
    else:
        combined_result, combined_path = train_or_load_run(
            combined_params,
            run_name=combined_run_name,
            iter_models_dir=iter_models_dir,
            examine_train_test_peaks=examine_train_test_peaks,
            skip_existing=skip_existing,
            trained_runs=trained_runs,
            description=f"Iter {iteration} combined best params",
        )
        combined_nse = mean_nse_from_result(combined_result)

    if combined_nse > previous_baseline_nse:
        print(
            f"\nCombined best params improved mean NSE "
            f"from {previous_baseline_nse:.4f} to {combined_nse:.4f}."
        )
        return BaselineUpdateDecision(
            should_continue=True,
            strategy="combined_best",
            baseline=combined_params,
            baseline_result=combined_result,
            baseline_model_path=combined_path,
            baseline_nse=combined_nse,
            previous_baseline_nse=previous_baseline_nse,
        )

    single_param_improvement = find_best_single_param_improvement(
        iteration_rows,
        baseline,
        previous_baseline_nse,
    )
    if single_param_improvement is not None:
        param_name, value_label, best_row, new_nse = single_param_improvement
        updated_baseline = dict(baseline)
        updated_baseline[param_name] = resolve_param_from_label(param_name, value_label)
        run_name = str(best_row["run_name"])
        if run_name in trained_runs:
            updated_result, updated_path = trained_runs[run_name]
        else:
            updated_result, updated_path = train_or_load_run(
                updated_baseline,
                run_name=run_name,
                iter_models_dir=iter_models_dir,
                examine_train_test_peaks=examine_train_test_peaks,
                skip_existing=skip_existing,
                trained_runs=trained_runs,
                description=f"Iter {iteration} single-param update ({param_name}={value_label})",
            )
        print(
            f"\nSingle-parameter update improved mean NSE via {param_name}={value_label}: "
            f"{previous_baseline_nse:.4f} -> {new_nse:.4f}."
        )
        return BaselineUpdateDecision(
            should_continue=True,
            strategy="single_param",
            baseline=updated_baseline,
            baseline_result=updated_result,
            baseline_model_path=updated_path,
            baseline_nse=new_nse,
            previous_baseline_nse=previous_baseline_nse,
            selected_param=param_name,
            selected_value_label=value_label,
        )

    print(
        f"\nNo NSE improvement found after iteration {iteration} "
        f"(baseline mean NSE={previous_baseline_nse:.4f}). Stopping."
    )
    return BaselineUpdateDecision(
        should_continue=False,
        strategy="stop",
        baseline=baseline,
        baseline_result=baseline_result,
        baseline_model_path=baseline_model_path,
        baseline_nse=previous_baseline_nse,
        previous_baseline_nse=previous_baseline_nse,
    )


def run_one_iteration(
    *,
    iteration: int,
    baseline: dict[str, Any],
    search_params: list[str],
    models_dir: Path,
    visuals_root: Path,
    comparison_metric: str,
    examine_train_test_peaks: bool,
    skip_existing: bool,
    baseline_result: RunResult | None = None,
    baseline_model_path: Path | None = None,
    highlight_baseline_in_boxplots: bool = DEFAULT_HIGHLIGHT_BASELINE_IN_BOXPLOTS,
) -> tuple[list[dict[str, Any]], RunResult, Path, dict[str, tuple[RunResult, Path]]]:
    """Execute one full one-at-a-time parameter sweep iteration.

    Trains or loads the baseline, then varies each search parameter across
    ``SEARCH_SPACE``, writes per-run visuals, aggregates comparison boxplots,
    and emits ``param_search_summary.csv`` and ``best_params.csv`` under
    ``visuals_root/iter_<iteration>/summaries/``.

    Args:
        iteration: 1-based iteration number (used in paths and logs).
        baseline: Starting hyperparameters for this iteration.
        search_params: Subset of keys to vary one-at-a-time.
        models_dir: Root directory for saved checkpoints.
        visuals_root: Root directory for plots and summary CSVs.
        comparison_metric: Metric for cross-value boxplots and error maps.
        examine_train_test_peaks: Peak evaluation spans train+test when True.
        skip_existing: Reuse checkpoints when present on disk.
        baseline_result: Optional pre-evaluated baseline to skip retraining.
        baseline_model_path: Checkpoint path paired with ``baseline_result``.
        highlight_baseline_in_boxplots: Highlight baseline in comparison plots.

    Returns:
        Tuple of ``(run_rows, baseline_result, baseline_model_path,
        trained_runs)`` where ``run_rows`` are summary dicts and
        ``trained_runs`` maps ``run_name`` to ``(RunResult, Path)``.

    Raises:
        ValueError: If a name in ``search_params`` is not in ``SEARCH_SPACE``.
    """
    iter_dir = visuals_root / f"iter_{iteration}"
    iter_models_dir = models_dir / f"iter_{iteration}"
    summaries_dir = iter_dir / "summaries"
    iter_dir.mkdir(parents=True, exist_ok=True)
    iter_models_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    trained_runs: dict[str, tuple[RunResult, Path]] = {}
    param_comparison_data: dict[str, dict[str, dict[str, float]]] = {}
    param_baseline_labels: dict[str, str] = {}
    baseline_run_name = build_run_name(BASELINE_NAME, baseline)

    if baseline_result is None or baseline_model_path is None:
        baseline_result, baseline_model_path = train_or_load_run(
            baseline,
            run_name=baseline_run_name,
            iter_models_dir=iter_models_dir,
            examine_train_test_peaks=examine_train_test_peaks,
            skip_existing=skip_existing,
            trained_runs=trained_runs,
            description=f"Iter {iteration} baseline",
        )
        write_run_visuals(
            baseline_result,
            run_name=baseline_run_name,
            visuals_root=iter_dir,
            varied_param="baseline",
            comparison_metric=comparison_metric,
            examine_train_test_peaks=examine_train_test_peaks,
        )
    else:
        trained_runs[baseline_run_name] = (baseline_result, baseline_model_path)
        print(
            f"\n=== Iter {iteration}: reusing baseline model from previous iteration "
            f"({baseline_run_name}) ==="
        )

    run_rows.append(
        summarize_run_row(
            run_name=baseline_run_name,
            varied_param="baseline",
            varied_value_label="baseline",
            params=baseline,
            errors_by_metric=baseline_result.errors_by_metric,
            return_period_values=baseline_result.return_period_values,
        )
    )

    for varied_param in search_params:
        if varied_param not in SEARCH_SPACE:
            raise ValueError(f"Unknown search parameter: {varied_param!r}")
        if varied_param == "weighted_adj" and params_have_several_gnn_layers(baseline):
            continue

        comparison_metric_by_value: dict[str, dict[str, float]] = {}
        baseline_value_label = param_display_label(baseline[varied_param])

        for option in SEARCH_SPACE[varied_param]:
            value, value_label = resolve_param_option(option)
            params = {**baseline, varied_param: option}
            run_name = build_run_name(BASELINE_NAME, params)

            if param_option_matches(baseline, varied_param, option):
                print(
                    f"\n=== Iter {iteration}: skipping {run_name} "
                    f"(varied: {varied_param}={value_label} matches baseline) ==="
                )
                result = baseline_result
            else:
                result, _ = train_or_load_run(
                    params,
                    run_name=run_name,
                    iter_models_dir=iter_models_dir,
                    examine_train_test_peaks=examine_train_test_peaks,
                    skip_existing=skip_existing,
                    trained_runs=trained_runs,
                    description=(
                        f"Iter {iteration} varied {varied_param}={value_label}"
                    ),
                )
                write_run_visuals(
                    result,
                    run_name=run_name,
                    visuals_root=iter_dir,
                    varied_param=varied_param,
                    comparison_metric=comparison_metric,
                    examine_train_test_peaks=examine_train_test_peaks,
                )

            run_row = summarize_run_row(
                run_name=run_name,
                varied_param=varied_param,
                varied_value_label=value_label,
                params=params,
                errors_by_metric=result.errors_by_metric,
                return_period_values=result.return_period_values,
            )
            run_rows.append(run_row)
            comparison_metric_by_value[value_label] = result.errors_by_metric.get(
                comparison_metric,
                {},
            )

        param_comparison_data[varied_param] = comparison_metric_by_value
        param_baseline_labels[varied_param] = baseline_value_label

    if param_comparison_data:
        shared_boxplot_ylim = compute_station_metric_boxplot_ylim(
            list(param_comparison_data.values()),
            ymin=UNIFIED_BOXPLOT_YMIN,
        )
        for varied_param, comparison_metric_by_value in param_comparison_data.items():
            comparison_path = (
                iter_dir
                / varied_param
                / f"{varied_param}_{comparison_metric}_boxplot_by_value.png"
            )
            summary_boxplot_path = (
                summaries_dir
                / f"{varied_param}_{comparison_metric}_boxplot_by_value.png"
            )
            plot_station_metric_boxplot_by_param_values(
                comparison_metric_by_value,
                error_metric=comparison_metric,
                varied_param=varied_param,
                output_path=comparison_path,
                show_plot=False,
                baseline_label=param_baseline_labels[varied_param],
                highlight_baseline=highlight_baseline_in_boxplots,
                format_labels=False,
            )
            plot_station_metric_boxplot_by_param_values(
                comparison_metric_by_value,
                error_metric=comparison_metric,
                varied_param=varied_param,
                output_path=summary_boxplot_path,
                show_plot=False,
                baseline_label=param_baseline_labels[varied_param],
                highlight_baseline=highlight_baseline_in_boxplots,
                ylim=shared_boxplot_ylim,
                format_labels=True,
            )

        grid_output_path = (
            summaries_dir / f"all_params_{comparison_metric}_boxplot_grid.png"
        )
        plot_station_metric_boxplot_grid(
            param_comparison_data,
            error_metric=comparison_metric,
            baseline_labels=param_baseline_labels,
            output_path=grid_output_path,
            show_plot=False,
            highlight_baseline=highlight_baseline_in_boxplots,
            ylim=shared_boxplot_ylim,
            format_labels=True,
            rows=4,
            fontsize=FONTSIZE_DEFAULT,
            show_outliers_as_dots=False,
        )
        print(f"Wrote iteration {iteration} comparison boxplots to {summaries_dir}")
        print(f"Wrote iteration {iteration} comparison boxplot grid to {grid_output_path}")

    summary_path = summaries_dir / "param_search_summary.csv"
    best_params_path = summaries_dir / "best_params.csv"
    write_summary_csv(run_rows, summary_path)
    write_best_params_csv(run_rows, search_params, baseline, best_params_path)
    print(f"\nWrote iteration {iteration} summary to {summary_path}")
    print(f"Wrote iteration {iteration} best params to {best_params_path}")
    return run_rows, baseline_result, baseline_model_path, trained_runs


def run_parameter_sweep(
    *,
    search_params: list[str] | None = None,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    visuals_root: str | Path = DEFAULT_VISUALS_ROOT,
    comparison_metric: str = DEFAULT_COMPARISON_METRIC,
    examine_train_test_peaks: bool = True,
    skip_existing: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    highlight_baseline_in_boxplots: bool = DEFAULT_HIGHLIGHT_BASELINE_IN_BOXPLOTS,
) -> pd.DataFrame:
    """Run the full iterative hyperparameter search until convergence or cap.

    Repeatedly calls :func:`run_one_iteration` and
    :func:`update_baseline_after_iteration`, writing per-iteration artifacts
    under ``visuals_root/iter_<n>/`` and checkpoints under
    ``models_dir/iter_<n>/``.

    Args:
        search_params: Parameters to sweep; defaults to ``SEARCH_PARAMS``.
        models_dir: Root directory for ``.pt`` checkpoints.
        visuals_root: Root for iteration folders and summary CSVs.
        comparison_metric: Metric for boxplots and error maps.
        examine_train_test_peaks: Include training period in peak metrics.
        skip_existing: Load existing checkpoints instead of retraining.
        max_iterations: Upper bound on iterative baseline-update rounds.
        highlight_baseline_in_boxplots: Highlight baseline in comparison plots.

    Returns:
        :class:`pandas.DataFrame` read from the latest
        ``param_search_summary.csv`` when available; otherwise a DataFrame
        built from all accumulated run rows (semicolon-separated columns).
    """
    models_dir = Path(models_dir)
    visuals_root = Path(visuals_root)
    models_dir.mkdir(parents=True, exist_ok=True)
    visuals_root.mkdir(parents=True, exist_ok=True)

    params_to_search = search_params or SEARCH_PARAMS
    baseline = dict(BASELINE)
    all_run_rows: list[dict[str, Any]] = []
    last_iteration = 0
    baseline_result: RunResult | None = None
    baseline_model_path: Path | None = None

    for iteration in range(1, max_iterations + 1):
        last_iteration = iteration
        iteration_rows, baseline_result, baseline_model_path, trained_runs = run_one_iteration(
            iteration=iteration,
            baseline=baseline,
            search_params=params_to_search,
            models_dir=models_dir,
            visuals_root=visuals_root,
            comparison_metric=comparison_metric,
            examine_train_test_peaks=examine_train_test_peaks,
            skip_existing=skip_existing,
            baseline_result=baseline_result,
            baseline_model_path=baseline_model_path,
            highlight_baseline_in_boxplots=highlight_baseline_in_boxplots,
        )
        all_run_rows.extend(iteration_rows)

        iter_models_dir = models_dir / f"iter_{iteration}"
        decision = update_baseline_after_iteration(
            baseline=baseline,
            baseline_result=baseline_result,
            baseline_model_path=baseline_model_path,
            iteration_rows=iteration_rows,
            search_params=params_to_search,
            iter_models_dir=iter_models_dir,
            examine_train_test_peaks=examine_train_test_peaks,
            skip_existing=skip_existing,
            trained_runs=trained_runs,
            iteration=iteration,
        )
        write_baseline_update_csv(
            decision,
            visuals_root / f"iter_{iteration}" / "summaries" / "baseline_update.csv",
        )

        baseline = decision.baseline
        baseline_result = decision.baseline_result
        baseline_model_path = decision.baseline_model_path

        if not decision.should_continue:
            break

    latest_summary = (
        visuals_root / f"iter_{last_iteration}" / "summaries" / "param_search_summary.csv"
    )
    if latest_summary.exists():
        return pd.read_csv(latest_summary, sep=";")
    return pd.DataFrame(all_run_rows)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the parameter search CLI.

    Returns:
        Parsed namespace with ``params``, ``models_dir``, ``visuals_dir``,
        ``comparison_metric``, ``max_iterations``, ``test_only_peaks``,
        ``skip_existing``, and ``no_highlight_baseline`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Iterative one-at-a-time hyperparameter sweep for TemporalGNN (GNN-LSTM). "
            "Map PNG export requires selenium and a browser driver; HTML maps are always saved."
        )
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help=f"Subset of parameters to sweep (default: {SEARCH_PARAMS}).",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory for saved models (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--visuals-dir",
        default=DEFAULT_VISUALS_ROOT,
        help=f"Root directory for sweep visuals (default: {DEFAULT_VISUALS_ROOT}).",
    )
    parser.add_argument(
        "--comparison-metric",
        default=DEFAULT_COMPARISON_METRIC,
        help=f"Metric used for cross-value boxplots (default: {DEFAULT_COMPARISON_METRIC}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum iterative sweep rounds (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--test-only-peaks",
        action="store_true",
        help="Evaluate return-period peaks on the test set only.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training when a model checkpoint already exists.",
    )
    parser.add_argument(
        "--no-highlight-baseline",
        action="store_true",
        help="Do not highlight the baseline value in parameter comparison boxplots.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse CLI args and run :func:`run_parameter_sweep`."""
    args = _parse_args()
    run_parameter_sweep(
        search_params=args.params,
        models_dir=args.models_dir,
        visuals_root=args.visuals_dir,
        comparison_metric=args.comparison_metric,
        examine_train_test_peaks=not args.test_only_peaks,
        skip_existing=args.skip_existing,
        max_iterations=args.max_iterations,
        highlight_baseline_in_boxplots=not args.no_highlight_baseline,
    )


if __name__ == "__main__":
    main()
