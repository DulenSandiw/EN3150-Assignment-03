"""
EN3150 Assignment 03 - Member 4
evaluation/benchmark.py

Benchmarks resource usage for all four models under identical hardware and
input conditions (same device, same 1 x 3 x 64 x 64 input):
  - total / trainable parameter counts
  - saved model size (MB), from the checkpoint file when one is available
  - estimated computational cost (MACs) via forward hooks on Conv2d/Linear
  - inference latency (mean/std, ms) and throughput (images/sec)

This script does NOT require the EuroSAT dataset - only the model
definitions - so parameter counts, MACs and latency can be benchmarked even
before a teammate's checkpoint file has been shared.

Usage:
    python evaluation/benchmark.py --model all
    python evaluation/benchmark.py --model model_b --variant sgd
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.registry import (  # noqa: E402
    MODEL_REGISTRY,
    build_model,
    resolve_checkpoint_path,
)

INPUT_SHAPE = (3, 64, 64)
DEFAULT_OUTPUT = PROJECT_DIR / "evaluation" / "results" / "benchmark.json"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_macs(model, input_shape=INPUT_SHAPE, device="cpu"):
    """Estimate multiply-accumulate operations (MACs) for one forward pass.

    Counts Conv2d (standard, depthwise and grouped) and Linear layers via
    forward hooks, since they dominate the compute cost of every model used
    in this project. Pooling/BatchNorm/activation layers are not counted
    (their FLOPs are negligible by comparison), so this is an estimate, not
    an exhaustive FLOP count from a profiler such as `thop`/`fvcore`.
    """
    macs = {"total": 0}
    hooks = []

    def conv_hook(module, _inputs, output):
        out_h, out_w = output.shape[-2], output.shape[-1]
        ops_per_position = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
        )
        macs["total"] += ops_per_position * module.out_channels * out_h * out_w

    def linear_hook(module, _inputs, _output):
        macs["total"] += module.in_features * module.out_features

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    model.eval()
    with torch.inference_mode():
        dummy = torch.randn(1, *input_shape, device=device)
        model(dummy)

    for hook in hooks:
        hook.remove()

    return macs["total"]


def checkpoint_size_mb(checkpoint_path):
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return None
    return Path(checkpoint_path).stat().st_size / (1024 ** 2)


def in_memory_size_mb(model):
    """Fallback model-size estimate (serialized state dict, in memory),
    used only when no checkpoint file is available on disk yet."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def benchmark_latency(model, input_shape=INPUT_SHAPE, device="cpu", batch_size=1, warmup=10, runs=100):
    model.eval()
    dummy = torch.randn(batch_size, *input_shape, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings_ms = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = sum(timings_ms) / len(timings_ms)
    variance = sum((t - mean_ms) ** 2 for t in timings_ms) / len(timings_ms)

    return {
        "batch_size": batch_size,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": variance ** 0.5,
        "throughput_images_per_sec": 1000.0 * batch_size / mean_ms,
    }


def benchmark_model(model_name, checkpoint_path, device, warmup=10, runs=100):
    checkpoint_exists = checkpoint_path is not None and Path(checkpoint_path).exists()

    model = build_model(
        model_name,
        checkpoint_path=checkpoint_path if checkpoint_exists else None,
        device=device,
    )

    # Apply the same freeze/unfreeze configuration each model was actually
    # trained with (e.g. MobileNetV2/EfficientNet-B0 only unfreeze their
    # last 2 blocks for fine-tuning), so "trainable parameters" reflects
    # the real training cost instead of a freshly-built model's full
    # capacity.
    configure_trainable = MODEL_REGISTRY[model_name].get("configure_trainable")
    if configure_trainable is not None:
        configure_trainable(model)

    total_params, trainable_params = count_parameters(model)
    macs = estimate_macs(model, device=device)
    latency = benchmark_latency(model, device=device, warmup=warmup, runs=runs)

    if checkpoint_exists:
        size_mb = checkpoint_size_mb(checkpoint_path)
        size_source = "checkpoint_file"
    else:
        size_mb = in_memory_size_mb(model)
        size_source = "untrained_state_dict_estimate"

    result = {
        "model": model_name,
        "display_name": MODEL_REGISTRY[model_name]["display_name"],
        "checkpoint_used": str(checkpoint_path) if checkpoint_exists else None,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "model_size_source": size_source,
        "estimated_macs": macs,
        "estimated_gmacs": macs / 1e9,
        "input_shape": list(INPUT_SHAPE),
        "device": str(device),
        **latency,
    }

    print(
        f"[{model_name}] params={total_params:,} (trainable={trainable_params:,}) "
        f"size={size_mb:.2f} MB [{size_source}] "
        f"MACs={macs / 1e6:.2f} M  "
        f"latency={latency['mean_latency_ms']:.3f}+/-{latency['std_latency_ms']:.3f} ms  "
        f"throughput={latency['throughput_images_per_sec']:.1f} img/s"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parameter counts, model size, MACs and inference "
                     "latency for all four models under consistent conditions."
    )
    parser.add_argument("--model", default="all", choices=list(MODEL_REGISTRY) + ["all"])
    parser.add_argument(
        "--variant",
        default=None,
        choices=["adam", "sgd", "momentum"],
        help="Only used for --model model_b: which optimizer checkpoint to benchmark.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the default checkpoint path (single-model mode only).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print("Benchmark device:", device)

    models_to_run = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    results = {}
    for model_name in models_to_run:
        checkpoint_path = resolve_checkpoint_path(
            model_name,
            variant=args.variant,
            override=args.checkpoint if args.model != "all" else None,
        )
        results[model_name] = benchmark_model(
            model_name, checkpoint_path, device, args.warmup, args.runs,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
