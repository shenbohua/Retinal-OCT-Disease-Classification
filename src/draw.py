"""
Visualisation module for retinal OCT classification project.

Data-driven plotting functions for:
- Learning curves (from DL training history CSVs)
- ROC curves (from DL prediction CSVs with probability scores)

All functions accept DataFrames or explicit paths and return matplotlib Figures.
Invoke via `python -m src.draw` or `python main.py draw-learning` / `draw-roc`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.config import CLASS_NAMES, make_paths

# ---------------------------------------------------------------------------
# matplotlib global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
LINE_STYLES = ["-", "--", "-.", ":"]
MODEL_COLORS = {
    "resnet18": "#e74c3c",
    "resnet34": "#3498db",
    "resnet50": "#2ecc71",
    "vgg16": "#f39c12",
    "mobilenet_v2": "#9b59b6",
}
MODEL_DISPLAY_NAMES = {
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "vgg16": "VGG16",
    "mobilenet_v2": "MobileNetV2",
}


# ===================================================================
# Data discovery
# ===================================================================


def discover_dl_runs() -> dict[str, dict[str, Path]]:
    """Scan outputs/runs/deeplearning/ and return {model: {history, predictions}}.

    Returns
    -------
    dict[str, dict[str, Path]]
        Keys are model names (resnet18, vgg16, ...).
        Each value is ``{"history": Path, "predictions": Path}``.
    """
    paths = make_paths()
    dl_root = paths.outputs_root / "runs" / "deeplearning"
    if not dl_root.exists():
        return {}

    runs: dict[str, dict[str, Path]] = {}
    for model_dir in sorted(dl_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        # walk into the deepest tables/ directory under val_final/*
        for cand in model_dir.rglob("tables"):
            history_path = cand / "history.csv"
            pred_path = cand / "predictions.csv"
            if history_path.exists() and pred_path.exists():
                runs[model_name] = {"history": history_path, "predictions": pred_path}
                break
    return runs


def discover_dl_history_paths() -> dict[str, Path]:
    """Return {model_name: path_to_history.csv} for all discovered DL runs."""
    runs = discover_dl_runs()
    return {name: info["history"] for name, info in runs.items()}


def discover_dl_prediction_paths() -> dict[str, Path]:
    """Return {model_name: path_to_predictions.csv} for all discovered DL runs."""
    runs = discover_dl_runs()
    return {name: info["predictions"] for name, info in runs.items()}


# ===================================================================
# Data loading helpers
# ===================================================================


def load_history(path: str | Path) -> pd.DataFrame:
    """Load a DL training history CSV.

    Expected columns: epoch, train_loss, val_loss, train_macro_f1, val_macro_f1,
    train_accuracy, val_accuracy.
    """
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    return df


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load a DL predictions CSV.

    Expected columns: filepath, y_true, y_pred, is_correct,
    prob_CNV, prob_DME, prob_DRUSEN, prob_NORMAL.
    """
    return pd.read_csv(path)


def load_histories_dict(
    sources: dict[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load multiple history files into {model_name: DataFrame}.

    If *sources* is None, auto-discovery is used.
    """
    if sources is None:
        sources = discover_dl_history_paths()
    return {name: load_history(p) for name, p in sources.items()}


def load_predictions_dict(
    sources: dict[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load multiple prediction files into {model_name: DataFrame}.

    If *sources* is None, auto-discovery is used.
    """
    if sources is None:
        sources = discover_dl_prediction_paths()
    return {name: load_predictions(p) for name, p in sources.items()}


# ===================================================================
# Learning curves
# ===================================================================


def plot_learning_curve(
    history_df: pd.DataFrame,
    model_name: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot loss and macro-F1 learning curves for a single model.

    Parameters
    ----------
    history_df : pd.DataFrame
        Must contain columns: epoch, train_loss, val_loss,
        train_macro_f1, val_macro_f1.
    model_name : str
        Display name for the figure title.
    save_path : Path or None
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = history_df["epoch"].values

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax_loss.plot(epochs, history_df["train_loss"], "o-", color="#2e86c1",
                 linewidth=1.5, markersize=4, label="Train Loss")
    ax_loss.plot(epochs, history_df["val_loss"], "s-", color="#e67e22",
                 linewidth=1.5, markersize=4, label="Val Loss")
    best_idx = history_df["val_loss"].idxmin()
    ax_loss.axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.4,
                    label=f"Best (epoch {epochs[best_idx]})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(frameon=False)
    ax_loss.grid(True, alpha=0.25)

    # --- Macro-F1 ---
    ax_f1.plot(epochs, history_df["train_macro_f1"], "o-", color="#2e86c1",
               linewidth=1.5, markersize=4, label="Train Macro-F1")
    ax_f1.plot(epochs, history_df["val_macro_f1"], "s-", color="#e67e22",
               linewidth=1.5, markersize=4, label="Val Macro-F1")
    best_idx = history_df["val_macro_f1"].idxmax()
    ax_f1.axvline(epochs[best_idx], color="red", linestyle="--", alpha=0.4,
                   label=f"Best (epoch {epochs[best_idx]})")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.set_title("Macro-F1")
    ax_f1.legend(frameon=False)
    ax_f1.grid(True, alpha=0.25)

    title = f"{model_name} — Learning Curves" if model_name else "Learning Curves"
    fig.suptitle(title, weight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_all_learning_curves(
    histories: dict[str, pd.DataFrame] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, plt.Figure]:
    """Generate per-model learning-curve figures for every model in *histories*.

    Also saves a combined overview figure.

    Parameters
    ----------
    histories : dict or None
        {model_name: history_DataFrame}.  Auto-discovered if None.
    save_dir : Path or None
        Directory for saved figures (default: outputs/figures).

    Returns
    -------
    dict[str, Figure]
        Mapping of model_name -> Figure (includes "combined_overview").
    """
    if histories is None:
        histories = load_histories_dict()

    if save_dir is None:
        save_dir = make_paths().figures_root
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, plt.Figure] = {}

    # Per-model figures
    for model_name, hist_df in histories.items():
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        save_path = save_dir / f"learning_curve_{model_name}.png"
        fig = plot_learning_curve(hist_df, model_name=display, save_path=save_path)
        figures[model_name] = fig

    # Combined overview: loss + macro-F1 across all models
    fig_overview, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(16, 6))
    for model_name, hist_df in histories.items():
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, "#333333")
        epochs = hist_df["epoch"].values
        ax_loss.plot(epochs, hist_df["val_loss"], linewidth=1.8, color=color, label=display)
        ax_f1.plot(epochs, hist_df["val_macro_f1"], linewidth=1.8, color=color, label=display)

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Val Loss"); ax_loss.set_title("Validation Loss")
    ax_loss.legend(frameon=False); ax_loss.grid(True, alpha=0.25)
    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Val Macro-F1"); ax_f1.set_title("Validation Macro-F1")
    ax_f1.legend(frameon=False); ax_f1.grid(True, alpha=0.25)
    fig_overview.suptitle("All Models — Validation Learning Curves", weight="bold", y=1.01)
    fig_overview.tight_layout()

    overview_path = save_dir / "learning_curve_all_models.png"
    fig_overview.savefig(overview_path)
    figures["combined_overview"] = fig_overview

    print(f"Saved {len(histories)} per-model learning curves + overview to {save_dir}")
    return figures


# ===================================================================
# ROC curves
# ===================================================================


def _compute_per_class_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute one-vs-rest ROC data for each class.

    Returns {class_name: {"fpr": array, "tpr": array, "auc": float}}.
    """
    n_classes = len(class_names)
    roc_data: dict[str, dict[str, np.ndarray]] = {}
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        roc_data[class_names[c]] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
    return roc_data


def plot_roc_curve(
    predictions_df: pd.DataFrame,
    model_name: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one-vs-rest ROC curves for a single model using real probability scores.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns: y_true, prob_CNV, prob_DME, prob_DRUSEN, prob_NORMAL.
    model_name : str
        Display name for the figure title.
    save_path : Path or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    missing = [c for c in prob_cols if c not in predictions_df.columns]
    if missing:
        raise KeyError(f"Prediction DataFrame missing probability columns: {missing}")

    y_true = predictions_df["y_true"].values
    y_prob = predictions_df[prob_cols].values.astype(np.float64)

    # Map class names to indices (0, 1, 2, 3)
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_true_idx = np.array([class_to_idx.get(str(t), -1) for t in y_true])
    if (y_true_idx < 0).any():
        raise ValueError("y_true contains unknown class labels.")

    roc_data = _compute_per_class_roc(y_true_idx, y_prob)

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([d["fpr"] for d in roc_data.values()]))
    mean_tpr = np.zeros_like(all_fpr)
    for d in roc_data.values():
        mean_tpr += np.interp(all_fpr, d["fpr"], d["tpr"])
    mean_tpr /= len(CLASS_NAMES)
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(8, 7))

    for c, cls_name in enumerate(CLASS_NAMES):
        d = roc_data[cls_name]
        ax.plot(d["fpr"], d["tpr"], linestyle=LINE_STYLES[c], color=CLASS_COLORS[c],
                linewidth=1.8, label=f"{cls_name} (AUC={d['auc']:.3f})")

    ax.plot(all_fpr, mean_tpr, "k-", linewidth=2.2,
            label=f"Macro-avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, linewidth=0.8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    title = f"{model_name} — ROC Curves (One-vs-Rest)" if model_name else "ROC Curves (One-vs-Rest)"
    ax.set_title(title, weight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_all_roc_curves(
    predictions: dict[str, pd.DataFrame] | None = None,
    save_dir: str | Path | None = None,
) -> dict[str, plt.Figure]:
    """Generate per-model ROC figures and a combined comparison figure.

    Parameters
    ----------
    predictions : dict or None
        {model_name: predictions_DataFrame}.  Auto-discovered if None.
    save_dir : Path or None

    Returns
    -------
    dict[str, Figure]
        Mapping of model_name -> Figure (includes "combined_comparison").
    """
    if predictions is None:
        predictions = load_predictions_dict()

    if save_dir is None:
        save_dir = make_paths().figures_root
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, plt.Figure] = {}

    # Per-model figures
    for model_name, pred_df in predictions.items():
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        save_path = save_dir / f"roc_{model_name}.png"
        try:
            fig = plot_roc_curve(pred_df, model_name=display, save_path=save_path)
            figures[model_name] = fig
        except KeyError as exc:
            print(f"  Skipping {model_name}: {exc}")

    # Combined comparison: one subplot per class, all model lines
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    fig_comb, axes = plt.subplots(2, 2, figsize=(14, 12))
    for c, cls_name in enumerate(CLASS_NAMES):
        ax = axes[c // 2][c % 2]
        for model_name, pred_df in predictions.items():
            if any(pc not in pred_df.columns for pc in prob_cols):
                continue
            display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            color = MODEL_COLORS.get(model_name, "#333333")
            y_true = pred_df["y_true"].values
            y_prob = pred_df[prob_cols].values.astype(np.float64)
            y_true_idx = np.array([class_to_idx.get(str(t), -1) for t in y_true])
            y_bin = (y_true_idx == c).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=1.6, color=color, label=f"{display} ({roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{cls_name}", weight="bold")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, alpha=0.25)

    fig_comb.suptitle("All Models — Per-Class ROC Comparison", weight="bold", y=1.01)
    fig_comb.tight_layout()

    comb_path = save_dir / "roc_all_models_comparison.png"
    fig_comb.savefig(comb_path)
    figures["combined_comparison"] = fig_comb

    print(f"Saved {len(figures) - 1} per-model ROC figures + combined comparison to {save_dir}")
    return figures


# ===================================================================
# Standalone entry point
# ===================================================================


def main() -> None:
    """Auto-discover all DL runs and generate learning-curve + ROC figures."""
    print("=" * 60)
    print("OCT Classification — Learning Curves & ROC Curves")
    print("=" * 60)

    run_info = discover_dl_runs()
    if not run_info:
        print("No DL run data found under outputs/runs/deeplearning/.")
        print("Run `python main.py train-dl` first to generate training data.")
        return

    print(f"\nDiscovered {len(run_info)} DL runs: {', '.join(run_info.keys())}")

    histories = {name: load_history(info["history"]) for name, info in run_info.items()}
    predictions = {name: load_predictions(info["predictions"]) for name, info in run_info.items()}

    print("\n--- Learning Curves ---")
    plot_all_learning_curves(histories)

    print("\n--- ROC Curves ---")
    plot_all_roc_curves(predictions)

    print("\nDone.")


if __name__ == "__main__":
    main()
