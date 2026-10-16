"""
EN3150 Assignment 03 - Member 4
Single source of truth describing how to build and load each of the group's
four models (Model A, Model B, MobileNetV2, EfficientNet-B0), so that
evaluate.py, benchmark.py and compare_models.py all treat every model the
same way.

This module intentionally knows nothing about training - it only builds
architectures (from the other members' `models/` package) and loads the
checkpoints they produced.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.model_a import StandardCNN  # noqa: E402
from models.model_b import LightweightCNN  # noqa: E402
from models.pretrained import (  # noqa: E402
    create_model,
    freeze_backbone,
    unfreeze_last_blocks,
)

SPLITS_CSV = PROJECT_DIR / "splits.csv"

# Custom CNNs (Model A/B) are trained on plain [0, 1] tensors (see dataset.py);
# the pretrained CNNs are fine-tuned on ImageNet-normalized input
# (see pretrained_data.py). Evaluation must reuse the same preprocessing
# each model was trained with, or the comparison would not be fair.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASIC_EVAL_TRANSFORM = transforms.ToTensor()
IMAGENET_EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_class_names(csv_path=SPLITS_CSV):
    """Derive the 10 EuroSAT class names, ordered by label_id, from the
    shared splits.csv. Reading it directly (instead of hardcoding the list
    in several files) enforces the group rule "use the same label mapping
    across all four models".
    """
    table = pd.read_csv(csv_path)
    mapping = table[["label_id", "label"]].drop_duplicates()
    mapping = mapping.sort_values("label_id")

    if mapping["label_id"].tolist() != list(range(len(mapping))):
        raise ValueError(
            "splits.csv label_id values are not a contiguous 0..N-1 range; "
            "refusing to guess the class ordering."
        )

    return mapping["label"].tolist()


MODEL_REGISTRY = {
    "model_a": {
        "display_name": "Model A (Standard CNN)",
        "build": lambda: StandardCNN(num_classes=10),
        "eval_transform": BASIC_EVAL_TRANSFORM,
        "default_checkpoint": "results/model_a_adam_01/best_model.pt",
    },
    "model_b": {
        "display_name": "Model B (Depthwise-Separable CNN)",
        "build": lambda: LightweightCNN(num_classes=10),
        "eval_transform": BASIC_EVAL_TRANSFORM,
        # Member 2 trained three optimizer variants; SGD+momentum scored
        # highest on the test set (see results/model_b_momentum_metrics.json)
        # so it is the default "Model B" checkpoint used in comparisons.
        "checkpoints": {
            "adam": "checkpoints/model_b_adam_best.pth",
            "sgd": "checkpoints/model_b_sgd_best.pth",
            "momentum": "checkpoints/model_b_momentum_best.pth",
        },
        "default_variant": "momentum",
        "default_checkpoint": "checkpoints/model_b_momentum_best.pth",
    },
    "mobilenet_v2": {
        "display_name": "MobileNetV2 (fine-tuned)",
        "build": lambda: create_model("mobilenet_v2", num_classes=10, pretrained=False),
        "eval_transform": IMAGENET_EVAL_TRANSFORM,
        "default_checkpoint": "outputs/mobilenet_v2/fine_tuning/best_model.pth",
        # Matches finetune_pretrained.py: backbone frozen, last 2 feature
        # blocks + classifier unfrozen for stage-2 fine-tuning.
        "configure_trainable": lambda model: _unfreeze_last_two_blocks(model),
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0 (fine-tuned)",
        "build": lambda: create_model("efficientnet_b0", num_classes=10, pretrained=False),
        "eval_transform": IMAGENET_EVAL_TRANSFORM,
        "default_checkpoint": "outputs/efficientnet_b0/fine_tuning/best_model.pth",
        # Matches finetune_efficientnet.py: same last-2-blocks scheme.
        "configure_trainable": lambda model: _unfreeze_last_two_blocks(model),
    },
}


def _unfreeze_last_two_blocks(model):
    """Reproduce the stage-2 fine-tuning freeze pattern used by
    finetune_pretrained.py / finetune_efficientnet.py, so that
    'trainable parameters' reported by the benchmark reflects what was
    actually updated during fine-tuning rather than a model's full
    capacity (every parameter looks "trainable" on a freshly-built,
    un-configured model otherwise).
    """
    freeze_backbone(model)
    unfreeze_last_blocks(model, num_blocks=2)
    return model


def resolve_checkpoint_path(model_name, variant=None, override=None):
    """Work out which checkpoint file to use for a model.

    `variant` only applies to model_b (adam/sgd/momentum); it is ignored for
    the other three models. `override` always wins when provided.
    """
    if override is not None:
        return Path(override)

    entry = MODEL_REGISTRY[model_name]

    if model_name == "model_b":
        variant = variant or entry["default_variant"]
        if variant not in entry["checkpoints"]:
            raise ValueError(
                f"Unknown model_b variant '{variant}'. "
                f"Choose from: {list(entry['checkpoints'])}"
            )
        return PROJECT_DIR / entry["checkpoints"][variant]

    return PROJECT_DIR / entry["default_checkpoint"]


def load_checkpoint_state_dict(checkpoint_path):
    """Every member's training script wraps the state dict in a checkpoint
    dict under the key 'model_state_dict' - this helper understands that
    shared convention for all four models.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def build_model(model_name, checkpoint_path=None, device="cpu"):
    """Construct a model and, if a checkpoint path is given, load its
    trained weights. Pass checkpoint_path=None to get an (untrained)
    architecture - useful for parameter counts and latency benchmarks that
    do not depend on the learned weights.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}"
        )

    model = MODEL_REGISTRY[model_name]["build"]()

    if checkpoint_path is not None:
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        model.load_state_dict(state_dict)

    return model.to(device)


CLASS_NAMES = load_class_names()
