
import torch
import torch.nn as nn
from pathlib import Path

from models.model_b import LightweightCNN
from training.data_adapter import load_eurosat_data


# Select GPU if available; otherwise use CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load EuroSAT training, validation and test data
train_loader, val_loader, test_loader = load_eurosat_data(
    batch_size=32
)

# Initialize Model B
model = LightweightCNN(num_classes=10).to(device)

# Cross-entropy loss for multiclass classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer (initial learning rate = 0.001)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Device:", device)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters()
        if p.requires_grad)
)



# Train the model for one epoch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # 1. Clear previous gradients
        optimizer.zero_grad()

        # 2. Forward propagation
        outputs = model(images)

        # 3. Calculate loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        # 6. Calculate training statistics
        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True
            )

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


def validate_one_epoch(model, loader, criterion, device):
    # Switch model to evaluation mode
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation only
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy




# Training configuration
NUM_EPOCHS = 20  # Initial test; increase later

best_val_acc = 0.0

# Store results for plotting graphs
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# Resume training from the latest checkpoint if available
start_epoch = 0

latest_checkpoint = Path("checkpoints/model_b_adam_latest.pth")

if latest_checkpoint.exists():
    checkpoint = torch.load(
        latest_checkpoint,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_acc = checkpoint["best_val_accuracy"]

    train_losses = checkpoint["train_losses"]
    train_accuracies = checkpoint["train_accuracies"]
    val_losses = checkpoint["val_losses"]
    val_accuracies = checkpoint["val_accuracies"]

    print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("Starting training from epoch 1")


for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}", flush=True)

    # Train on training images
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )


    # Evaluate on validation images
    val_loss, val_acc = validate_one_epoch(
        model, val_loader, criterion, device
    )

    
    # Save the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_model_path = checkpoint_dir / "model_b_adam_best.pth"

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_acc,
        }, best_model_path)

        print(
            f"New best model saved! "
            f"Validation Accuracy: {best_val_acc:.2f}%"
        )




    # Save metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    
    # Save progress after every completed epoch
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = checkpoint_dir / "model_b_adam_latest.pth"

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc,
    }, latest_checkpoint)

    print(f"Epoch {epoch + 1} checkpoint saved!")


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.2f}%"
    )
    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_acc:.2f}%"
    )

print("\nTraining completed!")


# Create directory for saved models
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save trained model and optimizer state
checkpoint_path = checkpoint_dir / "model_b_adam.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": len(train_losses),
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "val_losses": val_losses,
    "val_accuracies": val_accuracies,
}, checkpoint_path)

print(f"Model saved to: {checkpoint_path}")
