from __future__ import annotations

"""Manifest helpers for selecting train/eval rows without data leakage."""

from pathlib import Path

import pandas as pd


def select_split_rows(manifest_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return one split subset from split_manifest with explicit failure on empty."""
    selected = manifest_df[manifest_df["split_final"] == split_name].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split_final={split_name}.")
    return selected


def attach_data_paths(
    rows: pd.DataFrame,
    data_mode: str,
    processed_manifest_path: Path | None,
) -> pd.DataFrame:
    """
    Resolve actual file paths used by dataloaders.

    data_mode='raw' uses split_manifest.filepath directly.
    data_mode='processed' joins split_manifest with processed_manifest by source path.
    """
    mode = data_mode.lower()
    if mode not in {"raw", "processed"}:
        raise ValueError("data_mode must be one of {'raw','processed'}.")

    df = rows.copy()
    if mode == "raw":
        df["resolved_path"] = df["filepath"].astype(str)
        return df

    if processed_manifest_path is None or (not processed_manifest_path.exists()):
        raise FileNotFoundError(
            "Processed manifest not found. Run export-processed first or switch --data-mode raw. "
            f"Expected: {processed_manifest_path}"
        )

    pm = pd.read_csv(processed_manifest_path)
    required_cols = {"source_path", "processed_path", "split_final", "class_name"}
    missing = required_cols.difference(pm.columns)
    if missing:
        raise ValueError(
            f"Processed manifest is missing required columns: {sorted(missing)}. "
            f"File: {processed_manifest_path}"
        )

    merged = df.merge(
        pm[["source_path", "processed_path", "split_final", "class_name"]],
        how="left",
        left_on=["filepath", "split_final", "class_name"],
        right_on=["source_path", "split_final", "class_name"],
    )

    unresolved = int(merged["processed_path"].isna().sum())
    if unresolved > 0:
        raise ValueError(
            f"{unresolved} rows in split manifest were not found in processed manifest ({processed_manifest_path}). "
            "Re-run export-processed with matching splits/profile."
        )

    merged["resolved_path"] = merged["processed_path"].astype(str)
    return merged


def stratified_cap(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Cap rows using class-aware sampling while preserving label distribution."""
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df

    groups: list[pd.DataFrame] = []
    total = len(df)
    for _class_name, group in df.groupby("class_name"):
        n = max(1, int(round(len(group) / total * max_samples)))
        groups.append(group.sample(n=min(n, len(group)), random_state=seed))

    sampled = pd.concat(groups, axis=0)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    elif len(sampled) < max_samples:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            extra_n = min(max_samples - len(sampled), len(remaining))
            sampled = pd.concat([sampled, remaining.sample(n=extra_n, random_state=seed)], axis=0)

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
