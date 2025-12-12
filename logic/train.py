"""
Training script for deep learning models with MLFlow tracking.
This script trains models using transfer learning on the Oxford-IIIT Pet dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import mlflow
import mlflow.pytorch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
SEED = 42
EXPERIMENT_NAME = ""
MODEL_REGISTRY_NAME = ""
DATA_DIR = "./data"
PLOTS_DIR = "./plots"

# Training configurations to try
CONFIGS = [
    {
        "model_name": "mobilenet_v2",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 3,
    },
    {
        "model_name": "mobilenet_v2",
        "batch_size": 64,
        "learning_rate": 0.0001,
        "epochs": 3,
    },
    {
        "model_name": "mobilenet_v2",
        "batch_size": 32,
        "learning_rate": 0.0005,
        "epochs": 5,
    },
]


# ─────────────────────────────
# SETUP
# ─────────────────────────────
def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories():
    """Create necessary directories."""
    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(PLOTS_DIR).mkdir(exist_ok=True)


# ─────────────────────────────
# DATA PREPARATION
# ─────────────────────────────
def prepare_data(batch_size, train_split=0.8):
    """
    Prepare the Oxford-IIIT Pet dataset with train/validation split.

    Args:
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training (default 0.8)

    Returns:
        train_loader, val_loader, class_names
    """

    # Load dataset
    full_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=True, transform=transform
    )

    # Get class names
    class_names = full_dataset.classes

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, class_names


# ─────────────────────────────
# MODEL PREPARATION
# ─────────────────────────────
def prepare_model(model_name, num_classes):
    """
    Prepare a model for transfer learning.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes

    Returns:
        model: Prepared PyTorch model
    """
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze feature extractor
        for param in model.features.parameters():
            param.requires_grad = False

        # Modify classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


# ─────────────────────────────
# TRAINING
# ─────────────────────────────
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_curves(train_losses, val_losses, train_accs, val_accs, run_name):
    """Plot and save training curves."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = Path(PLOTS_DIR) / f"{run_name}_curves.png"
    plt.savefig(plot_path)
    plt.close()

    return str(plot_path)


def train_model(config, class_names):
    """
    Train a model with the given configuration and log to MLFlow.

    Args:
        config: Dictionary with training configuration
        class_names: List of class names
    """
    # Extract config
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    # Create run name
    run_name = f"{model_name}_bs{batch_size}_lr{learning_rate}_ep{epochs}"

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"{'='*60}\n")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, _ = prepare_data(batch_size)
    num_classes = len(class_names)

    # Prepare model
    model = prepare_model(model_name, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start MLFlow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "seed": SEED,
                "dataset": "OxfordIIITPet",
                "num_classes": num_classes,
                "train_split": 0.8,
                "image_size": 224,
            }
        )

        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Log metrics by epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )

        # Log final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": train_losses[-1],
                "final_train_accuracy": train_accs[-1],
                "final_val_loss": val_losses[-1],
                "final_val_accuracy": val_accs[-1],
            }
        )

        # Plot and log curves
        plot_path = plot_curves(
            train_losses, val_losses, train_accs, val_accs, run_name
        )
        mlflow.log_artifact(plot_path)

        # Save and log class labels
        class_labels_path = Path(PLOTS_DIR) / f"{run_name}_class_labels.json"
        with open(class_labels_path, "w", encoding='utf-8') as f:
            json.dump(class_names, f, indent=2)
        mlflow.log_artifact(str(class_labels_path), artifact_path="")

        # Log model
        mlflow.pytorch.log_model(
            model, artifact_path="model", registered_model_name=MODEL_REGISTRY_NAME
        )

        print(f"\nTraining completed for {run_name}")
        print(f"  Final Val Accuracy: {val_accs[-1]:.4f}\n")


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    """Main training function."""
    # Setup
    set_seed()
    setup_directories()

    # Set MLFlow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get class names (needed for all runs)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    temp_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", download=True, transform=transform
    )
    class_names = temp_dataset.classes

    print("Dataset: Oxford-IIIT Pet")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names[:5]}... (showing first 5)")

    # Train models with different configurations
    for config in CONFIGS:
        train_model(config, class_names)

    print("\n" + "=" * 60)
    print("All training completed!")
    print("Run 'mlflow ui' to view the results")
    print("=" * 60)


if __name__ == "__main__":
    main()