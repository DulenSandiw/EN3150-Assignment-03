
"""
EN3150 Assignment 03 - Member 3
Pretrained lightweight CNN architectures.

Models:
    1. MobileNetV2
    2. EfficientNet-B0

Both models are adapted for EuroSAT's 10 classes.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(model_name, num_classes=10, pretrained=True):
    """
    Create a lightweight CNN for EuroSAT classification.

    Args:
        model_name: 'mobilenet_v2' or 'efficientnet_b0'
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet weights

    Returns:
        PyTorch model
    """

    if model_name == "mobilenet_v2":

        weights = (
            models.MobileNet_V2_Weights.DEFAULT
            if pretrained else None
        )

        model = models.mobilenet_v2(weights=weights)

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "efficientnet_b0":

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )

        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model


def freeze_backbone(model):
    """
    Freeze feature extractor parameters.

    Only the classification head remains trainable.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model, num_blocks=2):
    """
    Unfreeze the last N feature blocks for fine-tuning.
    """

    for block in model.features[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def count_parameters(model):
    """
    Count total and trainable parameters.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


if __name__ == "__main__":

    for model_name in ["mobilenet_v2", "efficientnet_b0"]:

        model = create_model(model_name)

        freeze_backbone(model)

        total, trainable = count_parameters(model)

        print(f"\nModel: {model_name}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

        # Test required 64x64 input resolution.
        model.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            output = model(dummy_input)

        print("Output shape:", output.shape)
