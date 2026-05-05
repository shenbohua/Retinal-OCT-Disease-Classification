"""Batch runner for manifest-driven deep-learning experiments."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train all deep models with one shared protocol.")

    parser.add_argument("--manifest-path", type=str, default="outputs/tables/split_manifest.csv")
    parser.add_argument("--eval-split", choices=["val_final", "val_raw_holdout", "test_final"], default="val_final")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-mode", choices=["raw", "processed"], default="raw")
    parser.add_argument("--processed-profile", type=str, default="oct224_gray_png_v2")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)

    args = parser.parse_args()

    models = ["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet_v2"]

    for model_name in models:
        print("\n" + "=" * 70)
        print(f"Training model: {model_name}")
        print("=" * 70)

        command = [
            sys.executable,
            "-m",
            "src.deeplearning.train_deep",
            "--manifest-path",
            args.manifest_path,
            "--eval-split",
            args.eval_split,
            "--model", model_name,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--image-size", str(args.image_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--data-mode", args.data_mode,
            "--processed-profile", args.processed_profile,
            "--seed", str(args.seed),
        ]
        if args.max_train_samples is not None:
            command.extend(["--max-train-samples", str(args.max_train_samples)])
        if args.max_eval_samples is not None:
            command.extend(["--max-eval-samples", str(args.max_eval_samples)])
        if args.no_pretrained:
            command.append("--no-pretrained")
        if args.final_test:
            command.append("--final-test")

        if args.freeze_backbone:
            command.append("--freeze-backbone")

        subprocess.run(command, check=True)

    print("\nAll deep learning models finished training.")


if __name__ == "__main__":
    main()
