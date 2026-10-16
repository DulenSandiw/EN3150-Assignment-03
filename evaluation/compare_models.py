"""
EN3150 Assignment 03 - Member 4
evaluation/compare_models.py

Builds the final comparison tables:
  1. Model A  vs  Model B
  2. Model B  vs  MobileNetV2  vs  EfficientNet-B0

For each model it combines:
  - test accuracy / macro precision / macro recall / macro F1
      -> from this script's own evaluation/results/<model>/metrics.json if
         it has been (re)generated with evaluate.py, otherwise falls back
         to the metrics.json each member already committed.
  - parameter counts / model size / MACs / inference latency
      -> from evaluation/results/benchmark.json if present, otherwise
         benchmarked on the fly (does not require the dataset).
  - training time per epoch
      -> from each member's own training history file, when recorded.

Usage:
    python evaluation/compare_models.py
    python evaluation/compare_models.py --refresh-benchmark
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch  # noqa: E402

from evaluation.benchmark import benchmark_model  # noqa: E402
from evaluation.registry import MODEL_REGISTRY, resolve_checkpoint_path  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
BENCHMARK_JSON = RESULTS_DIR / "benchmark.json"

# Metrics.json files each member already committed, used as a fallback when
# evaluation/evaluate.py has not been (re)run for a given model (e.g. because
# the checkpoint file has not been shared on this machine yet).
LEGACY_METRICS = {
    "model_a": None,  # Member 1 has not produced a test-set evaluation yet.
    "model_b": {
        "adam": PROJECT_DIR / "results" / "model_b_adam_metrics.json",
        "sgd": PROJECT_DIR / "results" / "model_b_sgd_metrics.json",
        "momentum": PROJECT_DIR / "results" / "model_b_momentum_metrics.json",
    },
    "mobilenet_v2": PROJECT_DIR / "outputs" / "mobilenet_v2" / "evaluation" / "metrics.json",
    "efficientnet_b0": PROJECT_DIR / "outputs" / "efficientnet_b0" / "evaluation" / "metrics.json",
}


def _read_json(path):
    if path is None or not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_classification_metrics(model_name):
    """Prefer this pipeline's own evaluation output; fall back to whatever
    the responsible member already committed; otherwise report as missing.
    """
    own_metrics = _read_json(RESULTS_DIR / model_name / "metrics.json")
    if own_metrics is not None:
        own_metrics["source"] = "evaluation/evaluate.py"
        return own_metrics

    legacy = LEGACY_METRICS.get(model_name)

    if model_name == "model_b" and legacy is not None:
        # Pick whichever optimizer variant scored best on the test set.
        candidates = [(variant, _read_json(p)) for variant, p in legacy.items()]
        candidates = [(variant, m) for variant, m in candidates if m is not None]
        if not candidates:
            return None
        variant, metrics = max(candidates, key=lambda item: item[1]["test_accuracy"])
        metrics = dict(metrics)
        metrics["source"] = f"results/model_b_{variant}_metrics.json (best of 3 optimizers)"
        return metrics

    metrics = _read_json(legacy)
    if metrics is not None:
        metrics["source"] = str(legacy)
    return metrics


def load_benchmark_metrics(model_name, refresh):
    all_benchmarks = {} if refresh else (_read_json(BENCHMARK_JSON) or {})

    if model_name in all_benchmarks:
        return all_benchmarks[model_name]

    # Not benchmarked yet (or a refresh was requested) - run it now. This
    # only needs the model architecture, so it works even without the
    # dataset or a shared checkpoint.
    checkpoint_path = resolve_checkpoint_path(model_name)
    device = torch.device("cpu")
    return benchmark_model(model_name, checkpoint_path, device)


def load_training_time_per_epoch(model_name):
    """Average training seconds per epoch, when the member recorded it."""
    if model_name == "model_a":
        history_path = PROJECT_DIR / "results" / "model_a_adam_01" / "history.csv"
        if not history_path.exists():
            return None
        with open(history_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows or "train_seconds" not in rows[0]:
            return None
        seconds = [float(row["train_seconds"]) for row in rows]
        return sum(seconds) / len(seconds)

    # Member 2's and Member 3's training scripts do not currently record
    # per-epoch wall-clock time, so there is nothing to load for them yet.
    return None


def build_row(model_name, refresh_benchmark):
    classification = load_classification_metrics(model_name) or {}
    benchmark = load_benchmark_metrics(model_name, refresh_benchmark) or {}
    training_time = load_training_time_per_epoch(model_name)

    def pct(value):
        return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "N/A"

    def rnd(value, digits=2, suffix=""):
        return f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "N/A"

    return {
        "Model": MODEL_REGISTRY[model_name]["display_name"],
        "Test Accuracy": pct(classification.get("test_accuracy")),
        "Macro Precision": pct(classification.get("macro_precision")),
        "Macro Recall": pct(classification.get("macro_recall")),
        "Total Params": f"{benchmark.get('total_parameters', 'N/A'):,}"
            if isinstance(benchmark.get("total_parameters"), int) else "N/A",
        "Trainable Params": f"{benchmark.get('trainable_parameters', 'N/A'):,}"
            if isinstance(benchmark.get("trainable_parameters"), int) else "N/A",
        "Model Size (MB)": rnd(benchmark.get("model_size_mb"), 2),
        "Est. MACs (M)": rnd(benchmark.get("estimated_macs", 0) / 1e6 if benchmark.get("estimated_macs") else None, 2),
        "Latency (ms, batch=1)": rnd(benchmark.get("mean_latency_ms"), 3),
        "Throughput (img/s)": rnd(benchmark.get("throughput_images_per_sec"), 1),
        "Train Time/Epoch (s)": rnd(training_time, 1),
        "Metrics source": classification.get("source", "not evaluated yet"),
    }


def print_table(title, rows, columns):
    print(f"\n=== {title} ===")
    widths = {col: max(len(col), max(len(str(row[col])) for row in rows)) for col in columns}

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in columns))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the Model A vs Model B and Model B vs pretrained "
                     "comparison tables."
    )
    parser.add_argument(
        "--refresh-benchmark",
        action="store_true",
        help="Re-run parameter/MACs/latency benchmarking instead of reusing "
             "evaluation/results/benchmark.json.",
    )
    args = parser.parse_args()

    display_columns = [
        "Model", "Test Accuracy", "Macro Precision", "Macro Recall",
        "Total Params", "Trainable Params", "Model Size (MB)",
        "Est. MACs (M)", "Latency (ms, batch=1)", "Throughput (img/s)",
        "Train Time/Epoch (s)",
    ]
    csv_columns = display_columns + ["Metrics source"]

    rows = {
        model_name: build_row(model_name, args.refresh_benchmark)
        for model_name in MODEL_REGISTRY
    }

    custom_cnn_table = [rows["model_a"], rows["model_b"]]
    pretrained_table = [rows["model_b"], rows["mobilenet_v2"], rows["efficientnet_b0"]]

    print_table("Model A vs Model B", custom_cnn_table, display_columns)
    print_table("Model B vs MobileNetV2 vs EfficientNet-B0", pretrained_table, display_columns)

    for model_name, row in rows.items():
        if row["Metrics source"] == "not evaluated yet":
            checkpoint_path = resolve_checkpoint_path(model_name)
            reason = (
                "a checkpoint exists but has not been scored on the test set yet"
                if checkpoint_path.exists()
                else f"no checkpoint found at {checkpoint_path}"
            )
            print(
                f"\nWARNING: '{row['Model']}' has no test-set evaluation available "
                f"({reason}). Its accuracy/precision/recall columns are 'N/A' until "
                f"evaluation/evaluate.py is run for this model."
            )

    write_csv(RESULTS_DIR / "model_a_vs_model_b.csv", custom_cnn_table, csv_columns)
    write_csv(RESULTS_DIR / "model_b_vs_pretrained.csv", pretrained_table, csv_columns)


if __name__ == "__main__":
    main()
