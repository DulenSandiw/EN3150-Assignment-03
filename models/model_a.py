import torch
from torch import nn


class StandardCNN(nn.Module):
    """Standard CNN for 64x64 RGB images."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # [batch, 3, 64, 64] -> [batch, 16, 32, 32]
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # -> [batch, 32, 16, 16]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # -> [batch, 64, 8, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            # Average each feature map: -> [batch, 64, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = StandardCNN().to(device)

    # Dummy images check the architecture without training.
    dummy_images = torch.randn(32, 3, 64, 64, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(model)
    print("\nDevice:", device)
    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)
    print("Trainable parameters:", trainable_params)

    assert outputs.shape == (32, 10)
    assert trainable_params == 24234

    print("Model A architecture check passed!")