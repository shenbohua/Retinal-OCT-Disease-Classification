"""
Optional helper script.

This creates blank/template metric files so the group can see the expected format.

"""

from pathlib import Path
import pandas as pd

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

template_results = [
    {
        "feature": "end_to_end",
        "classifier": "vgg16",
        "eval_split": "val_final",
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "primary_metric_macro_f1": 0.0,
    },
    {
        "feature": "end_to_end",
        "classifier": "resnet50",
        "eval_split": "val_final",
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "primary_metric_macro_f1": 0.0,
    },
    {
        "feature": "end_to_end",
        "classifier": "mobilenet_v2",
        "eval_split": "val_final",
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "primary_metric_macro_f1": 0.0,
    }
]

for row in template_results:
    pd.DataFrame([row]).to_csv(results_dir / f"result_dl_{row['classifier']}_val_final.csv", index=False)

print("Created dummy metric files. Replace zeroes with actual model results after training.")
