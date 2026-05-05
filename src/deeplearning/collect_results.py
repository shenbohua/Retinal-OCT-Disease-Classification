"""Collect and validate traditional + deep-learning result tables."""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unify historical column naming into the coursework schema."""
    rename_map = {
        "model": "classifier",
        "type": "feature",
        "f1_macro": "macro_f1",
        "precision_macro": "macro_precision",
        "recall_macro": "macro_recall",
    }
    out = df.rename(columns=rename_map).copy()
    if "feature" not in out.columns:
        out["feature"] = "end_to_end"
    if "classifier" not in out.columns:
        out["classifier"] = ""
    if "eval_split" not in out.columns:
        out["eval_split"] = ""
    if "primary_metric_macro_f1" not in out.columns and "macro_f1" in out.columns:
        out["primary_metric_macro_f1"] = out["macro_f1"]
    return out


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def collect_results(results_dir: Path):
    result_files = sorted(results_dir.glob("result_*.csv"))
    metric_files = sorted(results_dir.glob("*_metrics.csv"))
    run_result_files = sorted((results_dir.parent / "runs").glob("**/tables/result.csv"))
    files = _dedupe_paths(result_files + metric_files + run_result_files)

    if not files:
        raise FileNotFoundError(
            f"No result files found in {results_dir}. "
            "Expected result_*.csv, *_metrics.csv, or outputs/runs/**/tables/result.csv."
        )

    frames = []

    for file in files:
        df = pd.read_csv(file)
        df = _normalise_columns(df)
        df["source_file"] = str(file.resolve().relative_to(results_dir.parent.resolve()))
        frames.append(df)

    all_results = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["macro_f1", "accuracy"] if c in all_results.columns]
    if sort_cols:
        all_results = all_results.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)

    preferred_cols = [
        "feature",
        "classifier",
        "eval_split",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "primary_metric_macro_f1",
        "train_time_sec",
        "inference_time_ms_per_image",
        "notes",
        "source_file"
    ]

    existing_cols = [c for c in preferred_cols if c in all_results.columns]
    remaining_cols = [c for c in all_results.columns if c not in existing_cols]

    all_results = all_results[existing_cols + remaining_cols]

    output_path = results_dir / "all_model_results.csv"
    all_results.to_csv(output_path, index=False)

    # Also save a Markdown table for easy pasting into the report.
    markdown_path = results_dir / "all_model_results_table.md"
    try:
        all_results.to_markdown(markdown_path, index=False)
    except Exception:
        # pandas.to_markdown may require optional dependency `tabulate`.
        markdown_path.write_text(
            "Install `tabulate` to enable Markdown table export.\n",
            encoding="utf-8",
        )

    # Validation report for protocol consistency.
    issues: list[dict[str, str | int]] = []
    allowed_splits = {"val_final", "val_raw_holdout", "test_final", ""}
    if "eval_split" in all_results.columns:
        bad = all_results[~all_results["eval_split"].astype(str).isin(allowed_splits)]
        for _, row in bad.iterrows():
            issues.append(
                {
                    "severity": "error",
                    "issue": "invalid_eval_split",
                    "source_file": str(row.get("source_file", "")),
                    "detail": f"eval_split={row.get('eval_split')}",
                }
            )

    required_numeric = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    for col in required_numeric:
        if col in all_results.columns:
            missing = all_results[all_results[col].isna()]
            for _, row in missing.iterrows():
                issues.append(
                    {
                        "severity": "warning",
                        "issue": f"missing_{col}",
                        "source_file": str(row.get("source_file", "")),
                        "detail": f"{col} is NaN",
                    }
                )

    test_rows = all_results[all_results.get("eval_split", "") == "test_final"] if "eval_split" in all_results.columns else pd.DataFrame()
    if len(test_rows) > 1:
        issues.append(
            {
                "severity": "warning",
                "issue": "multiple_test_final_rows",
                "source_file": "summary",
                "detail": f"Found {len(test_rows)} rows evaluated on test_final.",
            }
        )

    issue_df = pd.DataFrame(issues, columns=["severity", "issue", "source_file", "detail"])
    issue_path = results_dir / "all_model_results_validation.csv"
    issue_df.to_csv(issue_path, index=False)

    protocol_note_path = results_dir / "all_model_results_protocol_note.md"
    protocol_lines = [
        "# Result Protocol Check",
        "",
        f"- Total result rows: {len(all_results)}",
        f"- Rows on `val_final`: {int((all_results.get('eval_split') == 'val_final').sum()) if 'eval_split' in all_results.columns else 0}",
        f"- Rows on `val_raw_holdout`: {int((all_results.get('eval_split') == 'val_raw_holdout').sum()) if 'eval_split' in all_results.columns else 0}",
        f"- Rows on `test_final`: {int((all_results.get('eval_split') == 'test_final').sum()) if 'eval_split' in all_results.columns else 0}",
        f"- Validation issues: {len(issue_df)}",
    ]
    protocol_note_path.write_text("\n".join(protocol_lines), encoding="utf-8")

    return output_path, markdown_path, issue_path, protocol_note_path, all_results


def main():
    parser = argparse.ArgumentParser(description="Collect all model result CSV files.")
    parser.add_argument("--results_dir", type=str, default="outputs/tables")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    output_path, markdown_path, issue_path, protocol_note_path, all_results = collect_results(results_dir)

    print(f"Saved all model results to: {output_path}")
    print(f"Saved Markdown table to: {markdown_path}")
    print(f"Saved validation report to: {issue_path}")
    print(f"Saved protocol note to: {protocol_note_path}")
    print(all_results)


if __name__ == "__main__":
    main()
