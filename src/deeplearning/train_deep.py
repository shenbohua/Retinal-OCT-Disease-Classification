from __future__ import annotations

"""CLI wrapper for manifest-driven deep learning experiments."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import make_paths

from .run_experiment import DLProtocolError, run_dl_experiment
from .trainer import DLRunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deep learning models on OCT dataset.")
    parser.add_argument("--manifest-path", type=str, default="outputs/tables/split_manifest.csv")
    parser.add_argument("--eval-split", choices=["val_final", "val_raw_holdout", "test_final"], default="val_final")
    parser.add_argument("--final-test", action="store_true")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet_v2"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument("--data-mode", choices=["raw", "processed"], default="raw")
    parser.add_argument("--processed-profile", type=str, default="oct224_gray_png_v2")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)

    paths = make_paths()
    processed_manifest_path = paths.tables_root / f"processed_manifest_{args.processed_profile}.csv"

    cfg = DLRunConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        checkpoint_dir=paths.models_root,
        random_seed=args.seed,
        use_class_weight=(not args.no_class_weight),
        device=args.device,
        pretrained=(not args.no_pretrained),
        freeze_backbone=args.freeze_backbone,
        data_mode=args.data_mode,
        processed_profile=args.processed_profile,
        final_test=args.final_test,
    )

    try:
        result = run_dl_experiment(
            manifest_df=manifest_df,
            config=cfg,
            outputs_models_root=paths.models_root,
            outputs_tables_root=paths.tables_root,
            outputs_figures_root=paths.figures_root,
            processed_manifest_path=processed_manifest_path,
        )
    except DLProtocolError as exc:
        raise SystemExit(str(exc)) from exc

    out_path = paths.tables_root / f"result_dl_{args.model}_{args.eval_split}.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"Saved DL result row: {out_path}")


if __name__ == "__main__":
    main()
