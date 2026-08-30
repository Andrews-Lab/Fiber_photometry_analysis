"""FED3-aware FiPhoPHA draft.

Reads a FED3_FP_Combined.xlsx workbook created by FED3_post_processing.py,
keeps its time and metadata information, optionally downsamples the traces,
and runs metadata-labelled between-group bootstrap/permutation analyses.

This draft intentionally implements BETWEEN-GROUP analyses only.  A within-
subject mode should not be added until observations can be paired explicitly
by mouse/session/condition identifiers and the paired permutation method has
been validated.
"""

from __future__ import annotations

import argparse
import math
import threading
import traceback
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_EVENTS = ("Left", "Right", "Pellet", "Rewarded")
DEFAULT_CONFIDENCE_PERCENT = 95.0
DEFAULT_RESAMPLES = 1000
DEFAULT_PERMUTATIONS = 1000
DEFAULT_THRESHOLD_SECONDS = 0.5
DEFAULT_TARGET_RATE_HZ = 1.0
DEFAULT_RANDOM_SEED = 12345


@dataclass
class EventDataset:
    event: str
    source_sheet: str
    analysis_unit: str
    time: np.ndarray
    traces: pd.DataFrame
    metadata: pd.DataFrame


@dataclass
class AnalysisSettings:
    confidence_level: float = DEFAULT_CONFIDENCE_PERCENT / 100.0
    n_resamples: int = DEFAULT_RESAMPLES
    n_permutations: int = DEFAULT_PERMUTATIONS
    threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS
    baseline_start: float = -20.0
    baseline_end: float = 0.0
    comparison_start: float = 0.0
    comparison_end: float = 60.0
    trace_start: float = -20.0
    trace_end: float = 60.0
    downsample_factor: int = 1
    random_seed: int = DEFAULT_RANDOM_SEED


@dataclass
class ComparisonResult:
    event: str
    analysis_unit: str
    group_field: str
    stratify_field: str
    stratum_value: str
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    time: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    sem_a: np.ndarray
    sem_b: np.ndarray
    difference: np.ndarray
    difference_ci_low: np.ndarray
    difference_ci_high: np.ndarray
    group_a_ci_low: np.ndarray
    group_a_ci_high: np.ndarray
    group_b_ci_low: np.ndarray
    group_b_ci_high: np.ndarray
    raw_between_significant: np.ndarray
    thresholded_between_significant: np.ndarray
    group_a_from_zero: np.ndarray
    group_b_from_zero: np.ndarray
    baseline_mean_difference: float
    baseline_p_value: float
    comparison_mean_difference: float
    comparison_p_value: float
    threshold_points: int
    effective_threshold_seconds: float


def clean_text(value) -> str:
    """Normalise metadata values written by the FED3 post-processing workbook."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def available_events(workbook_path: str | Path) -> list[str]:
    sheet_names = set(pd.ExcelFile(workbook_path).sheet_names)
    return [event for event in EXPECTED_EVENTS if event in sheet_names]


def _read_combined_sheet(workbook_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Read a combined sheet while retaining its first-column row labels."""
    return pd.read_excel(workbook_path, sheet_name=sheet_name, index_col=0)


def load_event_dataset(
    workbook_path: str | Path,
    event: str,
    analysis_unit: str,
) -> EventDataset:
    if analysis_unit not in {"Individual events", "Animal event means"}:
        raise ValueError(f"Unknown analysis unit: {analysis_unit}")

    sheet_name = event if analysis_unit == "Individual events" else f"{event} EventMeans"
    table = _read_combined_sheet(workbook_path, sheet_name)

    if "Time (s)" not in table.columns:
        raise ValueError(f"Sheet '{sheet_name}' has no 'Time (s)' column.")
    if table.shape[0] < 2:
        raise ValueError(f"Sheet '{sheet_name}' does not contain metadata and signal rows.")

    # Detect every metadata row above the first row containing a numeric time.
    # The current FED3 exporter writes three such rows, but their labels are
    # user-editable and later versions may add more metadata fields.
    numeric_time = pd.to_numeric(table["Time (s)"], errors="coerce")
    numeric_positions = np.flatnonzero(numeric_time.notna().to_numpy())
    if len(numeric_positions) == 0:
        raise ValueError(f"Sheet '{sheet_name}' contains no numeric time values.")
    first_signal_position = int(numeric_positions[0])
    if first_signal_position == 0:
        raise ValueError(f"Sheet '{sheet_name}' contains no metadata rows above the signal.")
    metadata_block = table.iloc[:first_signal_position].copy()
    signal_block = table.iloc[first_signal_position:].copy()
    metadata_names = [clean_text(index) for index in metadata_block.index]
    if any(not name for name in metadata_names):
        raise ValueError(f"Sheet '{sheet_name}' contains an unnamed metadata row.")
    if len(set(metadata_names)) != len(metadata_names):
        raise ValueError(f"Sheet '{sheet_name}' contains duplicate metadata row names.")
    metadata_block.index = metadata_names

    observation_columns = [column for column in signal_block.columns if column != "Time (s)"]
    if not observation_columns:
        raise ValueError(f"Sheet '{sheet_name}' contains no observation columns.")

    time = pd.to_numeric(signal_block["Time (s)"], errors="coerce")
    traces = signal_block[observation_columns].apply(pd.to_numeric, errors="coerce")
    valid_time = time.notna()
    time = time.loc[valid_time].to_numpy(dtype=float)
    traces = traces.loc[valid_time].reset_index(drop=True)

    if len(time) < 2:
        raise ValueError(f"Sheet '{sheet_name}' has fewer than two valid time points.")
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"Time values in '{sheet_name}' are not strictly increasing.")
    if traces.columns.duplicated().any():
        duplicates = traces.columns[traces.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate observation names in '{sheet_name}': {duplicates}")

    metadata = metadata_block[observation_columns].T.copy()
    metadata.index = [clean_text(value) for value in metadata.index]
    metadata.index.name = "Observation"
    metadata.columns = metadata_names
    for column in metadata.columns:
        metadata[column] = metadata[column].map(clean_text)

    traces.columns = metadata.index
    completely_empty = traces.isna().all(axis=0)
    if completely_empty.any():
        empty_names = traces.columns[completely_empty].tolist()
        traces = traces.loc[:, ~completely_empty]
        metadata = metadata.drop(index=empty_names)

    return EventDataset(
        event=event,
        source_sheet=sheet_name,
        analysis_unit=analysis_unit,
        time=time,
        traces=traces,
        metadata=metadata,
    )


def sampling_interval(time: np.ndarray) -> float:
    differences = np.diff(np.asarray(time, dtype=float))
    if len(differences) == 0 or not np.all(np.isfinite(differences)):
        raise ValueError("Unable to calculate a sampling interval from the time vector.")
    interval = float(np.median(differences))
    if interval <= 0:
        raise ValueError("Sampling interval must be positive.")
    return interval


def suggest_downsample_factor(time: np.ndarray, target_rate_hz: float = DEFAULT_TARGET_RATE_HZ) -> int:
    if target_rate_hz <= 0:
        raise ValueError("Target sampling rate must be positive.")
    source_rate = 1.0 / sampling_interval(time)
    return max(1, int(round(source_rate / target_rate_hz)))


def downsample_dataset(dataset: EventDataset, factor: int) -> EventDataset:
    if factor < 1:
        raise ValueError("Downsample factor must be at least 1.")
    if factor == 1:
        return dataset

    n_rows = len(dataset.time)
    n_output_rows = n_rows // factor
    if n_output_rows < 2:
        raise ValueError("Downsample factor leaves fewer than two time points.")

    usable = n_output_rows * factor
    time = dataset.time[:usable].reshape(n_output_rows, factor).mean(axis=1)
    values = dataset.traces.iloc[:usable].to_numpy(dtype=float)
    values = np.nanmean(
        values.reshape(n_output_rows, factor, values.shape[1]),
        axis=1,
    )

    return EventDataset(
        event=dataset.event,
        source_sheet=dataset.source_sheet,
        analysis_unit=dataset.analysis_unit,
        time=time,
        traces=pd.DataFrame(values, columns=dataset.traces.columns),
        metadata=dataset.metadata.copy(),
    )


def trim_dataset(dataset: EventDataset, start_time: float, end_time: float) -> EventDataset:
    if start_time >= end_time:
        raise ValueError("Trace start must be less than trace end.")
    mask = (dataset.time >= start_time) & (dataset.time <= end_time)
    if mask.sum() < 2:
        raise ValueError(
            f"{dataset.event}: trace range {start_time:g} to {end_time:g} s "
            "contains fewer than two samples."
        )
    return EventDataset(
        event=dataset.event,
        source_sheet=dataset.source_sheet,
        analysis_unit=dataset.analysis_unit,
        time=dataset.time[mask],
        traces=dataset.traces.loc[mask].reset_index(drop=True),
        metadata=dataset.metadata.copy(),
    )


def threshold_points_from_seconds(time: np.ndarray, threshold_seconds: float) -> tuple[int, float]:
    """Convert duration to the nearest representable positive number of samples.

    Rounding mirrors the paper's convention (for example 0.5 s at about 10 Hz
    becomes 5 samples).  The effective duration is always reported because a
    heavily downsampled trace may not represent the requested duration closely.
    """
    if threshold_seconds < 0:
        raise ValueError("Consecutive threshold duration cannot be negative.")
    if threshold_seconds == 0:
        return 1, sampling_interval(time)
    dt = sampling_interval(time)
    points = max(1, int(round(threshold_seconds / dt)))
    return points, points * dt


def apply_consecutive_threshold(significance: np.ndarray, threshold_points: int) -> np.ndarray:
    significance = np.asarray(significance, dtype=bool)
    filtered = np.zeros_like(significance, dtype=bool)
    if threshold_points <= 1:
        return significance.copy()

    indices = np.flatnonzero(significance)
    if len(indices) == 0:
        return filtered
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.insert(indices[breaks + 1], 0, indices[0])
    ends = np.append(indices[breaks], indices[-1])
    for start, end in zip(starts, ends):
        if end - start + 1 >= threshold_points:
            filtered[start : end + 1] = True
    return filtered


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
    small_sample_adjustment: bool = True,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    sample_indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    bootstrap_means = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    if small_sample_adjustment and len(values) > 1:
        adjustment = math.sqrt(len(values) / (len(values) - 1))
        low *= adjustment
        high *= adjustment
    return float(low), float(high)


def _bootstrap_independent_difference_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan
    indices_a = rng.integers(0, len(values_a), size=(n_resamples, len(values_a)))
    indices_b = rng.integers(0, len(values_b), size=(n_resamples, len(values_b)))
    differences = values_a[indices_a].mean(axis=1) - values_b[indices_b].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(differences, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _window_permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    window_mask: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not np.any(window_mask):
        raise ValueError("A selected analysis window contains no downsampled time points.")

    values_a = np.nanmean(group_a[window_mask, :], axis=0)
    values_b = np.nanmean(group_b[window_mask, :], axis=0)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        raise ValueError("A selected analysis window contains no valid group observations.")

    observed = float(values_a.mean() - values_b.mean())
    pooled = np.concatenate([values_a, values_b])
    exceedances = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(pooled)
        difference = permuted[: len(values_a)].mean() - permuted[len(values_a) :].mean()
        exceedances += abs(difference) >= abs(observed)

    # Plus-one correction avoids reporting an impossible p=0 for random permutations.
    p_value = (exceedances + 1) / (n_permutations + 1)
    return observed, float(p_value)


def analyze_between_groups(
    dataset: EventDataset,
    group_field: str,
    group_a_label: str,
    group_b_label: str,
    settings: AnalysisSettings,
    comparison_seed: int,
    stratify_field: str = "",
    stratum_value: str = "All",
) -> ComparisonResult:
    if group_field not in dataset.metadata.columns:
        raise ValueError(f"Metadata field '{group_field}' is not available.")

    labels = dataset.metadata[group_field].map(clean_text)
    columns_a = labels.index[labels == group_a_label].tolist()
    columns_b = labels.index[labels == group_b_label].tolist()
    if len(columns_a) < 2 or len(columns_b) < 2:
        raise ValueError(
            f"{dataset.event}: '{group_a_label}' and '{group_b_label}' each need "
            "at least two observations."
        )

    group_a = dataset.traces[columns_a].to_numpy(dtype=float)
    group_b = dataset.traces[columns_b].to_numpy(dtype=float)
    time = dataset.time

    baseline_mask = (time >= settings.baseline_start) & (time < settings.baseline_end)
    comparison_mask = (time >= settings.comparison_start) & (time <= settings.comparison_end)
    if not np.any(baseline_mask):
        raise ValueError(f"{dataset.event}: baseline window contains no time points.")
    if not np.any(comparison_mask):
        raise ValueError(f"{dataset.event}: comparison window contains no time points.")

    threshold_points, effective_threshold = threshold_points_from_seconds(
        time, settings.threshold_seconds
    )
    n_timepoints = len(time)
    mean_a = np.nanmean(group_a, axis=1)
    mean_b = np.nanmean(group_b, axis=1)
    # Match FED3_post_processing.py's trace-overlay calculation exactly:
    # population SD across pooled observations divided by sqrt(total N).
    sem_a = np.nanstd(group_a, axis=1) / np.sqrt(group_a.shape[1])
    sem_b = np.nanstd(group_b, axis=1) / np.sqrt(group_b.shape[1])
    difference = mean_a - mean_b
    diff_low = np.full(n_timepoints, np.nan)
    diff_high = np.full(n_timepoints, np.nan)
    a_low = np.full(n_timepoints, np.nan)
    a_high = np.full(n_timepoints, np.nan)
    b_low = np.full(n_timepoints, np.nan)
    b_high = np.full(n_timepoints, np.nan)
    raw_between = np.zeros(n_timepoints, dtype=bool)
    a_from_zero = np.zeros(n_timepoints, dtype=bool)
    b_from_zero = np.zeros(n_timepoints, dtype=bool)

    rng = np.random.default_rng(comparison_seed)
    for index in range(n_timepoints):
        a_low[index], a_high[index] = _bootstrap_mean_ci(
            group_a[index, :], settings.n_resamples, settings.confidence_level, rng
        )
        b_low[index], b_high[index] = _bootstrap_mean_ci(
            group_b[index, :], settings.n_resamples, settings.confidence_level, rng
        )
        diff_low[index], diff_high[index] = _bootstrap_independent_difference_ci(
            group_a[index, :],
            group_b[index, :],
            settings.n_resamples,
            settings.confidence_level,
            rng,
        )
        raw_between[index] = (
            (diff_low[index] > 0 and diff_high[index] > 0)
            or (diff_low[index] < 0 and diff_high[index] < 0)
        )
        a_from_zero[index] = (
            (a_low[index] > 0 and a_high[index] > 0)
            or (a_low[index] < 0 and a_high[index] < 0)
        )
        b_from_zero[index] = (
            (b_low[index] > 0 and b_high[index] > 0)
            or (b_low[index] < 0 and b_high[index] < 0)
        )

    thresholded_between = apply_consecutive_threshold(raw_between, threshold_points)
    a_from_zero = apply_consecutive_threshold(a_from_zero, threshold_points)
    b_from_zero = apply_consecutive_threshold(b_from_zero, threshold_points)
    baseline_difference, baseline_p = _window_permutation_test(
        group_a, group_b, baseline_mask, settings.n_permutations, rng
    )
    comparison_difference, comparison_p = _window_permutation_test(
        group_a, group_b, comparison_mask, settings.n_permutations, rng
    )

    return ComparisonResult(
        event=dataset.event,
        analysis_unit=dataset.analysis_unit,
        group_field=group_field,
        stratify_field=stratify_field,
        stratum_value=stratum_value,
        group_a=group_a_label,
        group_b=group_b_label,
        n_a=len(columns_a),
        n_b=len(columns_b),
        time=time,
        mean_a=mean_a,
        mean_b=mean_b,
        sem_a=sem_a,
        sem_b=sem_b,
        difference=difference,
        difference_ci_low=diff_low,
        difference_ci_high=diff_high,
        group_a_ci_low=a_low,
        group_a_ci_high=a_high,
        group_b_ci_low=b_low,
        group_b_ci_high=b_high,
        raw_between_significant=raw_between,
        thresholded_between_significant=thresholded_between,
        group_a_from_zero=a_from_zero,
        group_b_from_zero=b_from_zero,
        baseline_mean_difference=baseline_difference,
        baseline_p_value=baseline_p,
        comparison_mean_difference=comparison_difference,
        comparison_p_value=comparison_p,
        threshold_points=threshold_points,
        effective_threshold_seconds=effective_threshold,
    )


def significant_intervals(result: ComparisonResult) -> list[dict]:
    indices = np.flatnonzero(result.thresholded_between_significant)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.insert(indices[breaks + 1], 0, indices[0])
    ends = np.append(indices[breaks], indices[-1])
    rows = []
    dt = sampling_interval(result.time)
    for start, end in zip(starts, ends):
        interval_difference = float(np.nanmean(result.difference[start : end + 1]))
        rows.append(
            {
                "Event": result.event,
                "Analysis Unit": result.analysis_unit,
                "Group Field": result.group_field,
                "Within Field": result.stratify_field or "None",
                "Within Value": result.stratum_value,
                "Comparison": f"{result.group_a} vs {result.group_b}",
                "Start Time (s)": float(result.time[start]),
                "End Time (s)": float(result.time[end]),
                "Approx Duration (s)": float((end - start + 1) * dt),
                "Mean Difference (A-B)": interval_difference,
                "Direction": result.group_a if interval_difference > 0 else result.group_b,
            }
        )
    return rows


def run_batch_analysis(
    workbook_path: str | Path,
    events: Iterable[str],
    analysis_unit: str,
    group_field: str,
    stratify_field: str | None,
    settings: AnalysisSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[ComparisonResult], list[EventDataset]]:
    progress = progress or (lambda _message: None)
    results: list[ComparisonResult] = []
    datasets: list[EventDataset] = []
    comparison_counter = 0

    for event in events:
        progress(f"Loading {event}...")
        dataset = load_event_dataset(workbook_path, event, analysis_unit)
        dataset = trim_dataset(dataset, settings.trace_start, settings.trace_end)
        dataset = downsample_dataset(dataset, settings.downsample_factor)
        datasets.append(dataset)
        if group_field not in dataset.metadata.columns:
            raise ValueError(
                f"Metadata field '{group_field}' is not present in {dataset.source_sheet}. "
                f"Available fields: {', '.join(dataset.metadata.columns)}"
            )
        if stratify_field and stratify_field not in dataset.metadata.columns:
            raise ValueError(
                f"Metadata field '{stratify_field}' is not present in {dataset.source_sheet}."
            )
        if stratify_field == group_field:
            raise ValueError("'Compare' and 'Within each' must use different metadata fields.")

        if stratify_field:
            stratum_values = sorted(
                value
                for value in dataset.metadata[stratify_field].map(clean_text).unique()
                if value
            )
        else:
            stratum_values = ["All"]

        for stratum_value in stratum_values:
            if stratify_field:
                stratum_columns = dataset.metadata.index[
                    dataset.metadata[stratify_field].map(clean_text) == stratum_value
                ].tolist()
                stratum_dataset = EventDataset(
                    event=dataset.event,
                    source_sheet=dataset.source_sheet,
                    analysis_unit=dataset.analysis_unit,
                    time=dataset.time,
                    traces=dataset.traces[stratum_columns],
                    metadata=dataset.metadata.loc[stratum_columns],
                )
            else:
                stratum_dataset = dataset

            group_values = sorted(
                value
                for value in stratum_dataset.metadata[group_field].map(clean_text).unique()
                if value
            )
            if len(group_values) < 2:
                progress(
                    f"Skipping {event}, {stratify_field or 'All'}={stratum_value}: "
                    f"fewer than two values for {group_field}."
                )
                continue

            for group_a, group_b in combinations(group_values, 2):
                progress(
                    f"Analyzing {event}, {stratify_field or 'All'}={stratum_value}: "
                    f"{group_a} vs {group_b}..."
                )
                comparison_seed = settings.random_seed + comparison_counter
                results.append(
                    analyze_between_groups(
                        stratum_dataset,
                        group_field,
                        group_a,
                        group_b,
                        settings,
                        comparison_seed,
                        stratify_field=stratify_field or "",
                        stratum_value=stratum_value,
                    )
                )
                comparison_counter += 1
    return results, datasets


def _safe_sheet_name(name: str, used: set[str]) -> str:
    invalid = set('[]:*?/\\')
    base = "".join("_" if char in invalid else char for char in name).strip() or "Sheet"
    base = base[:31]
    candidate = base
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = base[: 31 - len(suffix)] + suffix
        counter += 1
    used.add(candidate)
    return candidate


def export_results(
    output_path: str | Path,
    source_path: str | Path,
    results: list[ComparisonResult],
    datasets: list[EventDataset],
    settings: AnalysisSettings,
    color_map: dict[str, str] | None = None,
) -> Path:
    if not results:
        raise ValueError("No results were generated.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    interval_rows = []
    pointwise_rows = []
    significance_rows = []
    baseline_rows = []
    membership_rows = []

    for result in results:
        comparison = f"{result.group_a} vs {result.group_b}"
        summary_rows.append(
            {
                "Event": result.event,
                "Analysis Unit": result.analysis_unit,
                "Group Field": result.group_field,
                "Within Field": result.stratify_field or "None",
                "Within Value": result.stratum_value,
                "Comparison": comparison,
                f"N {result.group_a}": result.n_a,
                f"N {result.group_b}": result.n_b,
                "Baseline Mean Difference (A-B)": result.baseline_mean_difference,
                "Baseline P-Value": result.baseline_p_value,
                "Comparison Mean Difference (A-B)": result.comparison_mean_difference,
                "Comparison P-Value": result.comparison_p_value,
                "Significant Downsampled Points": int(result.thresholded_between_significant.sum()),
                "Consecutive Threshold (points)": result.threshold_points,
                "Effective Threshold (s)": result.effective_threshold_seconds,
            }
        )
        interval_rows.extend(significant_intervals(result))
        for index, time_value in enumerate(result.time):
            common = {
                "Event": result.event,
                "Analysis Unit": result.analysis_unit,
                "Group Field": result.group_field,
                "Within Field": result.stratify_field or "None",
                "Within Value": result.stratum_value,
                "Comparison": comparison,
                "Time (s)": float(time_value),
            }
            pointwise_rows.append(
                {
                    **common,
                    f"Mean {result.group_a}": result.mean_a[index],
                    f"SEM {result.group_a}": result.sem_a[index],
                    f"CI Low {result.group_a}": result.group_a_ci_low[index],
                    f"CI High {result.group_a}": result.group_a_ci_high[index],
                    f"Mean {result.group_b}": result.mean_b[index],
                    f"SEM {result.group_b}": result.sem_b[index],
                    f"CI Low {result.group_b}": result.group_b_ci_low[index],
                    f"CI High {result.group_b}": result.group_b_ci_high[index],
                    "Difference (A-B)": result.difference[index],
                    "Difference CI Low": result.difference_ci_low[index],
                    "Difference CI High": result.difference_ci_high[index],
                    "Raw Significant": int(result.raw_between_significant[index]),
                    "Thresholded Significant": int(result.thresholded_between_significant[index]),
                }
            )
            significance_rows.append(
                {
                    **common,
                    "Raw Significant": int(result.raw_between_significant[index]),
                    "Thresholded Significant": int(result.thresholded_between_significant[index]),
                }
            )
            baseline_rows.extend(
                [
                    {
                        "Event": result.event,
                        "Analysis Unit": result.analysis_unit,
                        "Group Field": result.group_field,
                        "Within Field": result.stratify_field or "None",
                        "Within Value": result.stratum_value,
                        "Group": result.group_a,
                        "Time (s)": float(time_value),
                        "Significant From Zero": int(result.group_a_from_zero[index]),
                    },
                    {
                        "Event": result.event,
                        "Analysis Unit": result.analysis_unit,
                        "Group Field": result.group_field,
                        "Within Field": result.stratify_field or "None",
                        "Within Value": result.stratum_value,
                        "Group": result.group_b,
                        "Time (s)": float(time_value),
                        "Significant From Zero": int(result.group_b_from_zero[index]),
                    },
                ]
            )

    seen_memberships = set()
    for dataset in datasets:
        for observation, row in dataset.metadata.iterrows():
            key = (dataset.event, dataset.analysis_unit, observation)
            if key in seen_memberships:
                continue
            seen_memberships.add(key)
            membership_rows.append(
                {
                    "Event": dataset.event,
                    "Analysis Unit": dataset.analysis_unit,
                    "Observation": observation,
                    **{column: clean_text(row[column]) for column in dataset.metadata.columns},
                }
            )

    first_dataset = datasets[0]
    dt_original_note = "Detected independently per event before downsampling"
    settings_rows = [
        {"Setting": "Source workbook", "Value": str(Path(source_path).resolve())},
        {"Setting": "Analysis type", "Value": "Between groups"},
        {"Setting": "Confidence level", "Value": settings.confidence_level},
        {"Setting": "Number of resamples", "Value": settings.n_resamples},
        {"Setting": "Number of permutations", "Value": settings.n_permutations},
        {"Setting": "Requested consecutive threshold (s)", "Value": settings.threshold_seconds},
        {"Setting": "Downsample factor", "Value": settings.downsample_factor},
        {"Setting": "Downsampled interval (s)", "Value": sampling_interval(first_dataset.time)},
        {"Setting": "Original sampling interval", "Value": dt_original_note},
        {"Setting": "Trace start (s, inclusive)", "Value": settings.trace_start},
        {"Setting": "Trace end (s, inclusive)", "Value": settings.trace_end},
        {"Setting": "Baseline start (s, inclusive)", "Value": settings.baseline_start},
        {"Setting": "Baseline end (s, exclusive)", "Value": settings.baseline_end},
        {"Setting": "Comparison start (s, inclusive)", "Value": settings.comparison_start},
        {"Setting": "Comparison end (s, inclusive)", "Value": settings.comparison_end},
        {"Setting": "Random seed", "Value": settings.random_seed},
        {
            "Setting": "Permutation p-value convention",
            "Value": "(exceedances + 1) / (permutations + 1)",
        },
    ]
    for group_name, color in sorted((color_map or {}).items()):
        settings_rows.append({"Setting": f"Plot colour - {group_name}", "Value": color})

    used_sheet_names: set[str] = set()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Analysis Summary", used_sheet_names)
        )
        pd.DataFrame(interval_rows).to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Significant Intervals", used_sheet_names)
        )
        pd.DataFrame(pointwise_rows).to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Pointwise Results", used_sheet_names)
        )
        pd.DataFrame(significance_rows).to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Significance Map", used_sheet_names)
        )
        pd.DataFrame(baseline_rows).drop_duplicates().to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Significance From Zero", used_sheet_names)
        )
        pd.DataFrame(membership_rows).to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Group Membership", used_sheet_names)
        )
        pd.DataFrame(settings_rows).to_excel(
            writer, index=False, sheet_name=_safe_sheet_name("Analysis Settings", used_sheet_names)
        )
    return output_path


def _boolean_runs(time: np.ndarray, values: np.ndarray) -> list[tuple[float, float]]:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.insert(indices[breaks + 1], 0, indices[0])
    ends = np.append(indices[breaks], indices[-1])
    dt = sampling_interval(time)
    return [(float(time[start]), float(time[end] + dt)) for start, end in zip(starts, ends)]


def _filename_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return token.strip("_") or "All"


def create_result_figures(
    output_workbook: str | Path,
    results: list[ComparisonResult],
    settings: AnalysisSettings,
    color_map: dict[str, str] | None = None,
    show_from_zero: bool = True,
    show_raw_significance: bool = False,
    save_svg: bool = False,
) -> Path:
    """Create one shared-time-axis significance/trace figure per comparison."""
    color_map = color_map or {}
    fallback_colors = list(plt.get_cmap("tab10").colors)
    output_workbook = Path(output_workbook)
    plot_folder = output_workbook.parent / "FED3_FiPhoPHA_Plots"
    plot_folder.mkdir(parents=True, exist_ok=True)

    for result in results:
        color_a = color_map.get(result.group_a, fallback_colors[0])
        color_b = color_map.get(result.group_b, fallback_colors[1])
        lane_count = 1 + int(show_raw_significance) + (2 if show_from_zero else 0)
        figure = plt.figure(figsize=(10.5, 7.2))
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=[max(1.4, lane_count * 0.48), 4.0],
            hspace=0.06,
        )
        significance_axis = figure.add_subplot(grid[0])
        trace_axis = figure.add_subplot(grid[1], sharex=significance_axis)

        lanes: list[tuple[str, np.ndarray, object]] = [
            (
                f"{result.group_a} vs {result.group_b}",
                result.thresholded_between_significant,
                "#111111",
            )
        ]
        if show_raw_significance:
            lanes.append(("Raw between-group", result.raw_between_significant, "#777777"))
        if show_from_zero:
            lanes.extend(
                [
                    (f"{result.group_a} vs zero", result.group_a_from_zero, color_a),
                    (f"{result.group_b} vs zero", result.group_b_from_zero, color_b),
                ]
            )

        y_positions = np.arange(len(lanes))[::-1]
        for y_position, (label, values, color) in zip(y_positions, lanes):
            for start, end in _boolean_runs(result.time, values):
                significance_axis.hlines(
                    y_position,
                    start,
                    end,
                    color=color,
                    linewidth=6,
                    capstyle="butt",
                )
        significance_axis.set_yticks(y_positions)
        significance_axis.set_yticklabels([lane[0] for lane in lanes])
        significance_axis.set_ylim(-0.65, len(lanes) - 0.35)
        significance_axis.spines[["top", "right", "left"]].set_visible(False)
        significance_axis.tick_params(axis="y", length=0)
        significance_axis.tick_params(axis="x", labelbottom=False)
        significance_axis.axvline(0, color="#555555", linestyle=":", linewidth=1.2)
        significance_axis.set_title(
            "Thresholded significance"
            + (" (plus optional lanes)" if len(lanes) > 1 else ""),
            loc="left",
            fontsize=11,
        )

        trace_axis.plot(result.time, result.mean_a, color=color_a, linewidth=2, label=result.group_a)
        trace_axis.fill_between(
            result.time,
            result.mean_a - result.sem_a,
            result.mean_a + result.sem_a,
            color=color_a,
            alpha=0.30,
            linewidth=0,
        )
        trace_axis.plot(result.time, result.mean_b, color=color_b, linewidth=2, label=result.group_b)
        trace_axis.fill_between(
            result.time,
            result.mean_b - result.sem_b,
            result.mean_b + result.sem_b,
            color=color_b,
            alpha=0.30,
            linewidth=0,
        )
        trace_axis.axhline(0, color="#555555", linestyle=":", linewidth=1.2)
        trace_axis.axvline(0, color="#555555", linestyle=":", linewidth=1.2)
        trace_axis.axvspan(
            settings.baseline_start,
            settings.baseline_end,
            color="#4C78A8",
            alpha=0.05,
            label="Baseline window",
        )
        trace_axis.axvspan(
            settings.comparison_start,
            settings.comparison_end,
            color="#F58518",
            alpha=0.04,
            label="Comparison window",
        )
        trace_axis.set_xlim(float(result.time.min()), float(result.time.max()))
        trace_axis.set_xlabel("Peri-event time (s)")
        trace_axis.set_ylabel("Z-score")
        trace_axis.spines[["top", "right"]].set_visible(False)
        trace_axis.legend(frameon=False, ncol=2, loc="upper right")
        trace_axis.text(
            0.01,
            0.02,
            f"Baseline p={result.baseline_p_value:.4g}   "
            f"Comparison p={result.comparison_p_value:.4g}",
            transform=trace_axis.transAxes,
            fontsize=9,
            va="bottom",
        )

        stratum_text = (
            f" | {result.stratify_field}: {result.stratum_value}"
            if result.stratify_field
            else ""
        )
        figure.suptitle(
            f"{result.event} | {result.group_a} vs {result.group_b}{stratum_text}\n"
            f"Mean ± SEM ({result.analysis_unit}); {result.threshold_points} point threshold "
            f"(~{result.effective_threshold_seconds:.3f} s)",
            fontsize=13,
            fontweight="bold",
        )
        figure.align_ylabels([significance_axis, trace_axis])
        figure.subplots_adjust(left=0.18, right=0.97, bottom=0.10, top=0.88)

        filename = "_".join(
            [
                _filename_token(result.event),
                _filename_token(result.stratum_value),
                _filename_token(result.group_a),
                "vs",
                _filename_token(result.group_b),
            ]
        )
        figure.savefig(plot_folder / f"{filename}.png", dpi=300, bbox_inches="tight")
        if save_svg:
            figure.savefig(plot_folder / f"{filename}.svg", bbox_inches="tight")
        plt.close(figure)

    return plot_folder


class FED3FiPhoPHAApp:
    def __init__(self, root: tk.Tk, initial_file: str | None = None):
        self.root = root
        self.root.title("FED3 FiPhoPHA — Between-Group Analysis")
        self._set_initial_geometry()

        self.file_var = tk.StringVar(value=initial_file or "")
        self.output_var = tk.StringVar()
        self.analysis_unit_var = tk.StringVar(value="Animal event means")
        self.group_field_var = tk.StringVar(value="Genotype")
        self.stratify_field_var = tk.StringVar(value="None")
        self.trace_start_var = tk.StringVar()
        self.trace_end_var = tk.StringVar()
        self.baseline_start_var = tk.StringVar()
        self.baseline_end_var = tk.StringVar(value="0")
        self.comparison_start_var = tk.StringVar(value="0")
        self.comparison_end_var = tk.StringVar()
        self.downsample_factor_var = tk.StringVar(value="1")
        self.confidence_var = tk.StringVar(value=str(DEFAULT_CONFIDENCE_PERCENT))
        self.resamples_var = tk.StringVar(value=str(DEFAULT_RESAMPLES))
        self.permutations_var = tk.StringVar(value=str(DEFAULT_PERMUTATIONS))
        self.threshold_seconds_var = tk.StringVar(value=str(DEFAULT_THRESHOLD_SECONDS))
        self.seed_var = tk.StringVar(value=str(DEFAULT_RANDOM_SEED))
        self.detected_var = tk.StringVar(value="Load a FED3_FP_Combined workbook.")
        self.threshold_preview_var = tk.StringVar(value="")
        self.create_plots_var = tk.BooleanVar(value=True)
        self.show_from_zero_var = tk.BooleanVar(value=True)
        self.show_raw_significance_var = tk.BooleanVar(value=False)
        self.save_svg_var = tk.BooleanVar(value=False)
        self.event_vars: dict[str, tk.BooleanVar] = {}
        self.metadata_fields: list[str] = []
        self.preview_dataset: EventDataset | None = None
        self.color_map: dict[str, str] = {}

        self._build_window()
        if initial_file:
            self.root.after(100, self.load_workbook)

    def _set_initial_geometry(self):
        """Fit the initial window inside smaller laptop displays."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1100, max(760, screen_width - 120))
        height = min(780, max(520, screen_height - 180))
        x_position = max(20, (screen_width - width) // 2)
        y_position = max(20, (screen_height - height) // 2)
        self.root.geometry(f"{width}x{height}+{x_position}+{y_position}")
        self.root.minsize(min(760, width), min(520, height))

    def _build_window(self):
        window_frame = ttk.Frame(self.root)
        window_frame.pack(fill="both", expand=True)
        window_frame.rowconfigure(0, weight=1)
        window_frame.columnconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(window_frame, highlightthickness=0)
        vertical_scrollbar = ttk.Scrollbar(
            window_frame,
            orient="vertical",
            command=self.scroll_canvas.yview,
        )
        self.scroll_canvas.configure(yscrollcommand=vertical_scrollbar.set)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")

        outer = ttk.Frame(self.scroll_canvas, padding=12)
        self.scroll_window = self.scroll_canvas.create_window(
            (0, 0), window=outer, anchor="nw"
        )
        outer.bind("<Configure>", self._update_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._resize_scroll_contents)
        # MouseWheel covers Windows and macOS; Button-4/5 covers Linux/X11.
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")

        outer.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(outer, text="1. Combined workbook", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew")
        input_frame.columnconfigure(1, weight=1)
        ttk.Label(input_frame, text="FED3_FP_Combined.xlsx").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.file_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        ttk.Button(input_frame, text="Load", command=self.load_workbook).grid(
            row=0, column=3, padx=(8, 0)
        )
        ttk.Label(input_frame, textvariable=self.detected_var, wraplength=920).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )

        design_frame = ttk.LabelFrame(outer, text="2. Events and grouping", padding=10)
        design_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        design_frame.columnconfigure(0, weight=1)
        self.events_container = ttk.Frame(design_frame)
        self.events_container.grid(row=0, column=0, columnspan=6, sticky="w")
        controls = ttk.Frame(design_frame)
        controls.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        ttk.Label(controls, text="Analysis unit").grid(row=0, column=0, sticky="w")
        unit_combo = ttk.Combobox(
            controls,
            textvariable=self.analysis_unit_var,
            values=["Animal event means", "Individual events"],
            state="readonly",
            width=25,
        )
        unit_combo.grid(row=0, column=1, sticky="w", padx=(8, 24))
        unit_combo.bind("<<ComboboxSelected>>", lambda _event: self.load_workbook())
        ttk.Label(controls, text="Compare").grid(row=0, column=2, sticky="w")
        self.group_combo = ttk.Combobox(
            controls,
            textvariable=self.group_field_var,
            values=[],
            state="readonly",
            width=22,
        )
        self.group_combo.grid(row=0, column=3, sticky="w", padx=(8, 24))
        self.group_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_preview())
        ttk.Label(controls, text="Within each").grid(row=0, column=4, sticky="w")
        self.stratify_combo = ttk.Combobox(
            controls,
            textvariable=self.stratify_field_var,
            values=["None"],
            state="readonly",
            width=22,
        )
        self.stratify_combo.grid(row=0, column=5, sticky="w", padx=(8, 0))
        self.stratify_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_preview())

        preview_columns = ("event", "within", "comparison", "n_a", "n_b")
        self.comparison_preview = ttk.Treeview(
            design_frame,
            columns=preview_columns,
            show="headings",
            height=4,
        )
        headings = {
            "event": "Event",
            "within": "Within",
            "comparison": "Comparison",
            "n_a": "N A",
            "n_b": "N B",
        }
        widths = {"event": 100, "within": 180, "comparison": 210, "n_a": 70, "n_b": 70}
        for column in preview_columns:
            self.comparison_preview.heading(column, text=headings[column])
            self.comparison_preview.column(column, width=widths[column], anchor="w")
        self.comparison_preview.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(10, 0))

        time_frame = ttk.LabelFrame(outer, text="3. Time and downsampling", padding=10)
        time_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        labels = [
            ("Trace start (s)", self.trace_start_var),
            ("Trace end (s)", self.trace_end_var),
            ("Baseline start (s)", self.baseline_start_var),
            ("Baseline end (s, exclusive)", self.baseline_end_var),
            ("Comparison start (s)", self.comparison_start_var),
            ("Comparison end (s)", self.comparison_end_var),
            ("Downsample factor", self.downsample_factor_var),
            ("Consecutive threshold (s)", self.threshold_seconds_var),
        ]
        for index, (label, variable) in enumerate(labels):
            row = index // 3
            column = (index % 3) * 2
            ttk.Label(time_frame, text=label).grid(row=row, column=column, sticky="w", pady=4)
            entry = ttk.Entry(time_frame, textvariable=variable, width=14)
            entry.grid(row=row, column=column + 1, sticky="w", padx=(6, 22), pady=4)
            entry.bind("<FocusOut>", lambda _event: self.update_threshold_preview())
        ttk.Label(time_frame, textvariable=self.threshold_preview_var, wraplength=920).grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(7, 0)
        )

        stats_frame = ttk.LabelFrame(outer, text="4. Statistical settings", padding=10)
        stats_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        stats = [
            ("Confidence (%)", self.confidence_var),
            ("Bootstrap resamples", self.resamples_var),
            ("Permutations", self.permutations_var),
            ("Random seed", self.seed_var),
        ]
        for index, (label, variable) in enumerate(stats):
            column = index * 2
            ttk.Label(stats_frame, text=label).grid(row=0, column=column, sticky="w")
            ttk.Entry(stats_frame, textvariable=variable, width=12).grid(
                row=0, column=column + 1, sticky="w", padx=(6, 22)
            )

        plot_frame = ttk.LabelFrame(outer, text="5. Figures", padding=10)
        plot_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(plot_frame, text="Create PNG figures", variable=self.create_plots_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(plot_frame, text="Show group-vs-zero lanes", variable=self.show_from_zero_var).grid(row=0, column=1, sticky="w", padx=(18, 0))
        ttk.Checkbutton(plot_frame, text="Show raw between-group lane", variable=self.show_raw_significance_var).grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Checkbutton(plot_frame, text="Also save SVG", variable=self.save_svg_var).grid(row=0, column=3, sticky="w", padx=(18, 0))
        ttk.Button(plot_frame, text="Choose group colours", command=self.choose_group_colours).grid(row=0, column=4, sticky="e", padx=(18, 0))

        output_frame = ttk.LabelFrame(outer, text="6. Output and run", padding=10)
        output_frame.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        outer.rowconfigure(5, weight=1)
        output_frame.columnconfigure(1, weight=1)
        output_frame.rowconfigure(2, weight=1)
        ttk.Label(output_frame, text="Results workbook").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.output_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=2)
        self.run_button = ttk.Button(output_frame, text="Run selected analyses", command=self.start_run)
        self.run_button.grid(row=0, column=3, padx=(8, 0))
        self.progress = ttk.Progressbar(output_frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 8))
        self.log = tk.Text(output_frame, height=9, wrap="word", state="disabled")
        self.log.grid(row=2, column=0, columnspan=4, sticky="nsew")

    def _update_scroll_region(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _resize_scroll_contents(self, event):
        # Keep the form as wide as the visible canvas while allowing it to be
        # taller than the window and therefore vertically scrollable.
        self.scroll_canvas.itemconfigure(self.scroll_window, width=event.width)

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return None
            # Windows normally reports multiples of 120; macOS commonly
            # reports smaller trackpad/wheel deltas.
            steps = int(-delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)

        widget_class = event.widget.winfo_class()
        if widget_class in {"TCombobox", "Listbox"}:
            return None
        if widget_class in {"Text", "Treeview"}:
            # Let a populated child control scroll until it reaches an edge.
            # Empty controls, or controls already at that edge, hand the wheel
            # event to the main form instead.
            try:
                first, last = event.widget.yview()
                child_can_scroll = (steps < 0 and first > 0.0) or (
                    steps > 0 and last < 1.0
                )
                if child_can_scroll:
                    return None
            except tk.TclError:
                pass

        self.scroll_canvas.yview_scroll(steps, "units")
        return "break"

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select FED3_FP_Combined workbook",
            filetypes=[("Excel workbooks", "*.xlsx")],
        )
        if path:
            self.file_var.set(path)
            self.load_workbook()

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save FED3 FiPhoPHA results",
            defaultextension=".xlsx",
            filetypes=[("Excel workbooks", "*.xlsx")],
        )
        if path:
            self.output_var.set(path)

    def _set_event_controls(self, events: list[str]):
        existing = {event: variable.get() for event, variable in self.event_vars.items()}
        for child in self.events_container.winfo_children():
            child.destroy()
        self.event_vars = {}
        ttk.Label(self.events_container, text="Events:").grid(row=0, column=0, sticky="w")
        for index, event in enumerate(events, start=1):
            variable = tk.BooleanVar(value=existing.get(event, True))
            self.event_vars[event] = variable
            ttk.Checkbutton(
                self.events_container,
                text=event,
                variable=variable,
                command=self.refresh_preview,
            ).grid(
                row=0, column=index, sticky="w", padx=(12, 0)
            )

    def load_workbook(self):
        try:
            path = Path(self.file_var.get())
            if not path.is_file():
                raise ValueError("Select an existing FED3_FP_Combined.xlsx workbook.")
            events = available_events(path)
            if not events:
                raise ValueError("No Left, Right, Pellet, or Rewarded sheets were found.")
            self._set_event_controls(events)
            dataset = load_event_dataset(path, events[0], self.analysis_unit_var.get())
            self.preview_dataset = dataset
            self.metadata_fields = list(dataset.metadata.columns)
            self.group_combo.configure(values=self.metadata_fields)
            self.stratify_combo.configure(values=["None"] + self.metadata_fields)
            if self.group_field_var.get() not in self.metadata_fields:
                preferred = "Genotype" if "Genotype" in self.metadata_fields else self.metadata_fields[0]
                self.group_field_var.set(preferred)
            if self.stratify_field_var.get() not in ["None"] + self.metadata_fields:
                self.stratify_field_var.set("None")

            dt = sampling_interval(dataset.time)
            rate = 1.0 / dt
            factor = suggest_downsample_factor(dataset.time)
            self.downsample_factor_var.set(str(factor))
            self.trace_start_var.set(f"{dataset.time.min():.6g}")
            self.trace_end_var.set(f"{dataset.time.max():.6g}")
            self.baseline_start_var.set(f"{dataset.time.min():.6g}")
            self.baseline_end_var.set("0")
            self.comparison_start_var.set("0")
            self.comparison_end_var.set(f"{dataset.time.max():.6g}")
            self.detected_var.set(
                f"Detected {', '.join(events)}; {len(dataset.time)} time points from "
                f"{dataset.time.min():.3f} to {dataset.time.max():.3f} s; "
                f"sampling rate {rate:.3f} Hz. Suggested factor {factor} gives about "
                f"{rate / factor:.3f} Hz. Metadata: {', '.join(self.metadata_fields)}."
            )
            if not self.output_var.get():
                self.output_var.set(str(path.with_name(path.stem + "_FiPhoPHA.xlsx")))
            self.update_threshold_preview()
            self.refresh_preview()
            self.write_log("Workbook loaded successfully.")
        except Exception as error:
            messagebox.showerror("Workbook error", str(error))

    def refresh_preview(self):
        for item in self.comparison_preview.get_children():
            self.comparison_preview.delete(item)
        path = Path(self.file_var.get())
        if not path.is_file() or not self.event_vars:
            return
        group_field = self.group_field_var.get()
        stratify_field = self.stratify_field_var.get()
        if stratify_field == "None":
            stratify_field = ""
        if group_field == stratify_field:
            self.comparison_preview.insert(
                "", "end", values=("—", "—", "Compare and Within must differ", "—", "—")
            )
            return
        try:
            for event, variable in self.event_vars.items():
                if not variable.get():
                    continue
                dataset = load_event_dataset(path, event, self.analysis_unit_var.get())
                if group_field not in dataset.metadata.columns:
                    continue
                if stratify_field:
                    strata = sorted(
                        value
                        for value in dataset.metadata[stratify_field].map(clean_text).unique()
                        if value
                    )
                else:
                    strata = ["All"]
                for stratum in strata:
                    metadata = dataset.metadata
                    if stratify_field:
                        metadata = metadata[
                            metadata[stratify_field].map(clean_text) == stratum
                        ]
                    levels = sorted(
                        value for value in metadata[group_field].map(clean_text).unique() if value
                    )
                    for group_a, group_b in combinations(levels, 2):
                        n_a = int((metadata[group_field].map(clean_text) == group_a).sum())
                        n_b = int((metadata[group_field].map(clean_text) == group_b).sum())
                        within_text = f"{stratify_field}: {stratum}" if stratify_field else "All"
                        self.comparison_preview.insert(
                            "",
                            "end",
                            values=(event, within_text, f"{group_a} vs {group_b}", n_a, n_b),
                        )
        except Exception as error:
            self.write_log(f"Comparison preview unavailable: {error}")

    def choose_group_colours(self):
        if self.preview_dataset is None:
            messagebox.showinfo("Colours", "Load a workbook first.")
            return
        group_field = self.group_field_var.get()
        if group_field not in self.preview_dataset.metadata.columns:
            return
        values = sorted(
            value
            for value in self.preview_dataset.metadata[group_field].map(clean_text).unique()
            if value
        )
        fallback = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]
        for index, value in enumerate(values):
            current = self.color_map.get(value, fallback[index % len(fallback)])
            selected = colorchooser.askcolor(
                color=current,
                title=f"Choose colour for {group_field}: {value}",
                parent=self.root,
            )[1]
            if selected:
                self.color_map[value] = selected
        self.write_log(
            "Group colours: "
            + ", ".join(f"{value}={self.color_map.get(value, 'default')}" for value in values)
        )

    def update_threshold_preview(self):
        if self.preview_dataset is None:
            return
        try:
            factor = int(self.downsample_factor_var.get())
            threshold_seconds = float(self.threshold_seconds_var.get())
            downsampled = downsample_dataset(self.preview_dataset, factor)
            points, effective = threshold_points_from_seconds(
                downsampled.time, threshold_seconds
            )
            self.threshold_preview_var.set(
                f"The requested {threshold_seconds:g} s threshold becomes {points} consecutive "
                f"downsampled point(s), an effective duration of approximately {effective:.3f} s."
            )
        except Exception as error:
            self.threshold_preview_var.set(f"Threshold preview unavailable: {error}")

    def _settings_from_window(self) -> AnalysisSettings:
        confidence = float(self.confidence_var.get()) / 100.0
        settings = AnalysisSettings(
            confidence_level=confidence,
            n_resamples=int(self.resamples_var.get()),
            n_permutations=int(self.permutations_var.get()),
            threshold_seconds=float(self.threshold_seconds_var.get()),
            baseline_start=float(self.baseline_start_var.get()),
            baseline_end=float(self.baseline_end_var.get()),
            comparison_start=float(self.comparison_start_var.get()),
            comparison_end=float(self.comparison_end_var.get()),
            trace_start=float(self.trace_start_var.get()),
            trace_end=float(self.trace_end_var.get()),
            downsample_factor=int(self.downsample_factor_var.get()),
            random_seed=int(self.seed_var.get()),
        )
        if not 0 < settings.confidence_level < 1:
            raise ValueError("Confidence must be between 0 and 100 percent.")
        if settings.n_resamples < 2 or settings.n_permutations < 1:
            raise ValueError("Use at least 2 resamples and 1 permutation.")
        if settings.baseline_start >= settings.baseline_end:
            raise ValueError("Baseline start must be less than baseline end.")
        if settings.comparison_start >= settings.comparison_end:
            raise ValueError("Comparison start must be less than comparison end.")
        if settings.trace_start >= settings.trace_end:
            raise ValueError("Trace start must be less than trace end.")
        if settings.baseline_start < settings.trace_start or settings.baseline_end > settings.trace_end:
            raise ValueError("Baseline window must fall within the trace range.")
        if settings.comparison_start < settings.trace_start or settings.comparison_end > settings.trace_end:
            raise ValueError("Comparison window must fall within the trace range.")
        if settings.downsample_factor < 1:
            raise ValueError("Downsample factor must be at least 1.")
        return settings

    def start_run(self):
        try:
            input_path = Path(self.file_var.get())
            output_path = Path(self.output_var.get())
            events = [event for event, variable in self.event_vars.items() if variable.get()]
            if not input_path.is_file():
                raise ValueError("Load a valid combined workbook first.")
            if not events:
                raise ValueError("Select at least one event.")
            if not output_path.name:
                raise ValueError("Choose an output workbook.")
            settings = self._settings_from_window()
            group_field = self.group_field_var.get()
            stratify_field = self.stratify_field_var.get()
            stratify_field = None if stratify_field == "None" else stratify_field
            if stratify_field == group_field:
                raise ValueError("'Compare' and 'Within each' must use different fields.")
            analysis_unit = self.analysis_unit_var.get()
            create_plots = self.create_plots_var.get()
            show_from_zero = self.show_from_zero_var.get()
            show_raw_significance = self.show_raw_significance_var.get()
            save_svg = self.save_svg_var.get()
            color_map = self.color_map.copy()
        except Exception as error:
            messagebox.showerror("Settings error", str(error))
            return

        self.run_button.configure(state="disabled")
        self.progress.start(10)
        self.write_log("Starting analysis...")

        def worker():
            try:
                results, datasets = run_batch_analysis(
                    input_path,
                    events,
                    analysis_unit,
                    group_field,
                    stratify_field,
                    settings,
                    progress=lambda message: self.root.after(0, self.write_log, message),
                )
                export_results(
                    output_path,
                    input_path,
                    results,
                    datasets,
                    settings,
                    color_map=color_map,
                )
                if create_plots:
                    plot_folder = create_result_figures(
                        output_path,
                        results,
                        settings,
                        color_map=color_map,
                        show_from_zero=show_from_zero,
                        show_raw_significance=show_raw_significance,
                        save_svg=save_svg,
                    )
                    self.root.after(0, self.write_log, f"Figures saved to {plot_folder}")
                self.root.after(0, self._run_complete, output_path, None)
            except Exception as error:
                details = traceback.format_exc()
                self.root.after(0, self._run_complete, None, f"{error}\n\n{details}")

        threading.Thread(target=worker, daemon=True).start()

    def _run_complete(self, output_path: Path | None, error: str | None):
        self.progress.stop()
        self.run_button.configure(state="normal")
        if error:
            self.write_log(error)
            messagebox.showerror("Analysis failed", error.split("\n\n", 1)[0])
        else:
            self.write_log(f"Finished. Results saved to {output_path}")
            messagebox.showinfo("Analysis complete", f"Results saved to:\n{output_path}")

    def write_log(self, message: str):
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def launch_gui(initial_file: str | None = None):
    root = tk.Tk()
    FED3FiPhoPHAApp(root, initial_file=initial_file)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="FED3_FP_Combined.xlsx path")
    parser.add_argument("--output", help="Results .xlsx path")
    parser.add_argument("--events", nargs="+", default=list(EXPECTED_EVENTS))
    parser.add_argument("--unit", choices=["Animal event means", "Individual events"], default="Animal event means")
    parser.add_argument("--group-by", default="Genotype")
    parser.add_argument("--within-each", default="None")
    parser.add_argument("--trace-start", type=float)
    parser.add_argument("--trace-end", type=float)
    parser.add_argument("--baseline-start", type=float)
    parser.add_argument("--baseline-end", type=float, default=0.0)
    parser.add_argument("--comparison-start", type=float, default=0.0)
    parser.add_argument("--comparison-end", type=float)
    parser.add_argument("--downsample-factor", type=int)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_PERCENT)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--threshold-seconds", type=float, default=DEFAULT_THRESHOLD_SECONDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-svg", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.no_gui:
        launch_gui(args.input)
        return
    if not args.input or not args.output:
        raise SystemExit("--input and --output are required with --no-gui")

    events_present = available_events(args.input)
    events = [event for event in args.events if event in events_present]
    if not events:
        raise SystemExit("None of the requested event sheets are present.")
    preview = load_event_dataset(args.input, events[0], args.unit)
    factor = args.downsample_factor or suggest_downsample_factor(preview.time)
    settings = AnalysisSettings(
        confidence_level=args.confidence / 100.0,
        n_resamples=args.resamples,
        n_permutations=args.permutations,
        threshold_seconds=args.threshold_seconds,
        baseline_start=args.baseline_start if args.baseline_start is not None else float(preview.time.min()),
        baseline_end=args.baseline_end,
        comparison_start=args.comparison_start,
        comparison_end=args.comparison_end if args.comparison_end is not None else float(preview.time.max()),
        trace_start=args.trace_start if args.trace_start is not None else float(preview.time.min()),
        trace_end=args.trace_end if args.trace_end is not None else float(preview.time.max()),
        downsample_factor=factor,
        random_seed=args.seed,
    )
    results, datasets = run_batch_analysis(
        args.input,
        events,
        args.unit,
        args.group_by,
        None if args.within_each == "None" else args.within_each,
        settings,
        progress=print,
    )
    output = export_results(args.output, args.input, results, datasets, settings)
    if not args.no_plots:
        plot_folder = create_result_figures(
            output,
            results,
            settings,
            show_from_zero=True,
            save_svg=args.save_svg,
        )
        print(f"Figures saved to {plot_folder}")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
