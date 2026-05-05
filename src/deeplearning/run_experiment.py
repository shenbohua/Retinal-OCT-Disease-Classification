from __future__ import annotations

"""Manifest-driven DL experiment runner aligned with coursework split protocol."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES
from src.evaluate import compute_metrics, per_class_table, save_confusion_matrix
from src.utils import timed

from .dataset_manifest import attach_data_paths, select_split_rows, stratified_cap
from .trainer import DLRunConfig, DLTrainer


class DLProtocolError(ValueError):
    """Raised when command arguments violate the evaluation protocol."""


def run_dl_experiment(
    manifest_df: pd.DataFrame,
    config: DLRunConfig,
    outputs_models_root: Path,
    outputs_tables_root: Path,
    outputs_figures_root: Path,
    processed_manifest_path: Path | None = None,
) -> dict[str, float | str]:
    """Run one DL experiment and save report-ready artifacts."""
    if config.eval_split == "test_final" and not config.final_test:
        raise DLProtocolError(
            "`test_final` is reserved for final reporting. Use --final-test to explicitly confirm."
        )

    train_rows = select_split_rows(manifest_df, "train_final")
    eval_rows = select_split_rows(manifest_df, config.eval_split)

    train_rows = attach_data_paths(train_rows, config.data_mode, processed_manifest_path)
    eval_rows = attach_data_paths(eval_rows, config.data_mode, processed_manifest_path)

    train_rows = stratified_cap(train_rows, config.max_train_samples, config.random_seed)
    eval_rows = stratified_cap(eval_rows, config.max_eval_samples, config.random_seed)

    train_paths = train_rows["resolved_path"].astype(str).tolist()
    eval_paths = eval_rows["resolved_path"].astype(str).tolist()
    train_classes = train_rows["class_name"].tolist()
    eval_classes = eval_rows["class_name"].tolist()

    trainer = DLTrainer(config=config)
    model = trainer.make_model()
    optimizer = trainer.make_optimizer(model)

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_train_idx = np.array([class_to_idx[c] for c in train_classes], dtype=np.int64)
    class_weights = (
        trainer.class_weight_from_labels(y_train_idx, num_classes=len(CLASS_NAMES))
        if config.use_class_weight
        else None
    )
    criterion = trainer.make_criterion(class_weights=class_weights)

    train_loader = trainer.make_loader(
        filepaths=train_paths,
        class_names=train_classes,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=True,
        shuffle=True,
    )
    eval_loader = trainer.make_loader(
        filepaths=eval_paths,
        class_names=eval_classes,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=False,
        shuffle=False,
    )

    outputs_models_root.mkdir(parents=True, exist_ok=True)
    outputs_tables_root.mkdir(parents=True, exist_ok=True)
    outputs_figures_root.mkdir(parents=True, exist_ok=True)

    run_name = f"dl_{config.model_name}_{config.eval_split}"
    run_dir = (
        outputs_tables_root.parent
        / "runs"
        / "deeplearning"
        / config.model_name
        / config.eval_split
        / f"seed{config.random_seed}_img{config.image_size}_{config.data_mode}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "models" / "best.pt"
    config_dump = json.loads(json.dumps(asdict(config), default=str))
    history_rows: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0

    with timed() as train_timer:
        for epoch in range(1, config.epochs + 1):
            train_metrics = trainer.train_one_epoch(model, train_loader, optimizer, criterion)
            eval_metrics = trainer.validate(model, eval_loader, criterion)
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": eval_metrics["loss"],
                    "val_macro_f1": eval_metrics["macro_f1"],
                    "val_accuracy": eval_metrics["accuracy"],
                }
            )
            if eval_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = eval_metrics["macro_f1"]
                best_epoch = epoch
                trainer.save_checkpoint(
                    state={
                        "epoch": epoch,
                        "model_name": config.model_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_macro_f1": best_macro_f1,
                        "config": config_dump,
                    },
                    path=checkpoint_path,
                )

    # Load best checkpoint before final export.
    import torch

    ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    with timed() as infer_timer:
        y_true_idx, y_pred_idx, y_probs, pred_paths = trainer.predict(model, eval_loader)

    y_true = np.array([CLASS_NAMES[i] for i in y_true_idx])
    y_pred = np.array([CLASS_NAMES[i] for i in y_pred_idx])
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    infer_ms_per_image = (infer_timer["seconds"] / len(y_true) * 1000.0) if len(y_true) else 0.0

    history_path = run_dir / "tables" / "history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    # Compatibility flat file.
    pd.DataFrame(history_rows).to_csv(outputs_tables_root / f"history_{run_name}.csv", index=False)

    per_class_df = per_class_table(y_true=y_true, y_pred=y_pred, labels=CLASS_NAMES)
    per_class_path = run_dir / "tables" / "per_class.csv"
    per_class_df.to_csv(per_class_path, index=False)
    # Compatibility flat file.
    per_class_df.to_csv(outputs_tables_root / f"per_class_{run_name}.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "filepath": pred_paths,
            "y_true": y_true,
            "y_pred": y_pred,
            "is_correct": (y_true == y_pred),
        }
    )
    for i, class_name in enumerate(CLASS_NAMES):
        pred_df[f"prob_{class_name}"] = y_probs[:, i] if len(y_probs) else []

    pred_path = run_dir / "tables" / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    # Compatibility flat file.
    pred_df.to_csv(outputs_tables_root / f"predictions_{run_name}.csv", index=False)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=CLASS_NAMES,
        path=run_dir / "figures" / "confusion.png",
        title=f"Confusion Matrix: {run_name}",
    )
    # Compatibility flat file.
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=CLASS_NAMES,
        path=outputs_figures_root / f"confusion_{run_name}.png",
        title=f"Confusion Matrix: {run_name}",
    )

    result_row = {
        "feature": "end_to_end",
        "classifier": config.model_name,
        "eval_split": config.eval_split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "primary_metric_macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "train_time_sec": train_timer["seconds"],
        "inference_time_ms_per_image": infer_ms_per_image,
        "notes": (
            f"best_epoch={best_epoch};batch_size={config.batch_size};"
            f"image_size={config.image_size};data_mode={config.data_mode};"
            f"pretrained={config.pretrained};freeze_backbone={config.freeze_backbone}"
        ),
        "model_path": str(checkpoint_path.resolve()),
        "predictions_path": str(pred_path.resolve()),
        "history_path": str(history_path.resolve()),
        "run_dir": str(run_dir.resolve()),
    }
    pd.DataFrame([result_row]).to_csv(run_dir / "tables" / "result.csv", index=False)
    log_path = outputs_tables_root / "deeplearning_experiment_log.csv"
    if log_path.exists():
        old_df = pd.read_csv(log_path)
        pd.concat([old_df, pd.DataFrame([result_row])], ignore_index=True).to_csv(log_path, index=False)
    else:
        pd.DataFrame([result_row]).to_csv(log_path, index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "model": config.model_name,
                "eval_split": config.eval_split,
                "seed": config.random_seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "data_mode": config.data_mode,
                "processed_profile": config.processed_profile,
                "pretrained": config.pretrained,
                "freeze_backbone": config.freeze_backbone,
                "use_class_weight": config.use_class_weight,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result_row
