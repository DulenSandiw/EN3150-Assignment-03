
import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=1,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(16, 32),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(64, 96),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_trainable_parameters(model):
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":
    model = LightweightCNN()
    model.eval()

    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))

    count = count_trainable_parameters(model)

    print("Output shape:", output.shape)
    print("Trainable parameters:", count)

    assert output.shape == (1, 10)
    assert count <= 100_000
    print("Model B verification successful!")
