"""
Training script with observability enabled.

This script trains a CNN on Fashion-MNIST while sending layer-wise metrics
to a monitoring server at http://localhost:8000
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import time

# Import the monitoring module
from nn_monitor import create_monitor

# Import dataset loader
from dataset_loader import get_torch_loaders


# Step 1: Define neural network models
class MinimalFashionCNN(nn.Module):
    """Minimal CNN with single conv layer - baseline for comparison."""
    def __init__(self):
        super(MinimalFashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 13 * 13, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DecentFashionCNN(nn.Module):
    """A decent CNN with 4 layers for better Fashion-MNIST accuracy.

    Architecture:
    - Conv2D(32) -> ReLU
    - Conv2D(64) -> ReLU -> MaxPool(2x2)
    - Conv2D(128) -> ReLU -> MaxPool(2x2)
    - Flatten -> Dense(128) -> ReLU -> Dense(10)
    """
    def __init__(self):
        super(DecentFashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 26x26 -> 24x24
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 12x12 -> 10x10
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 5 * 5, 128)  # After 2x pooling: 5x5x128
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))      # 26x26x32
        x = self.pool(self.relu(self.conv2(x)))  # 12x12x64
        x = self.pool(self.relu(self.conv3(x)))  # 5x5x128
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_default_layer_groups(model_class):
    """Get default layer groups for a model class.

    Returns a dict mapping group names to lists of layer IDs.
    Layer IDs use '/' separator instead of '.' for JSON compatibility.
    """
    if model_class == DecentFashionCNN:
        return {
            "feature_extractor": [
                "conv1",
                "conv2",
                "conv3"
            ],
            "classifier": [
                "fc1",
                "fc2"
            ]
        }
    elif model_class == MinimalFashionCNN:
        return {
            "feature_extractor": [
                "conv1"
            ],
            "classifier": [
                "fc"
            ]
        }
    return None


def train_model(model_class=DecentFashionCNN, num_epochs=5,
                monitor_endpoint="http://localhost:8000/api/v1/metrics/layerwise",
                log_interval=10, batch_size=64, learning_rate=0.001,
                layer_groups=None):
    """Train the model with monitoring enabled."""

    # Step 1: Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader, test_loader = get_torch_loaders(
        batch_size=batch_size,
        root='data',
        transform=transform,
        download=True,
        num_workers=0,
        pin_memory=False,
    )

    # Step 2: Create model
    model = model_class()

    # Use default layer groups if not provided
    if layer_groups is None:
        layer_groups = get_default_layer_groups(model_class)
        if layer_groups:
            print(f"[Monitor] Using default layer groups: {list(layer_groups.keys())}")

    # Step 3: Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            major, minor = props.major, props.minor
            # Allow GPUs with compute capability >= 6 (Pascal and newer)
            if major < 6:
                print(f"GPU compute capability {major}.{minor} is too old. Falling back to CPU.")
                device = torch.device("cpu")
        except Exception:
            print("Could not query CUDA device properties; using CPU instead.")
            device = torch.device("cpu")

    model = model.to(device)
    print(f"Using device: {device}")

    # Step 4: Create monitor (before optimizer so hooks are registered)
    run_id = f"{model_class.__name__}_{time.strftime('%Y%m%d_%H%M%S')}"
    monitor = create_monitor(
        model=model,
        api_endpoint=monitor_endpoint,
        run_id=run_id,
        log_interval=log_interval,
        batch_size=batch_size,
        layer_groups=layer_groups
    )

    # Step 5: Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 6: Training loop
    print(f"\nStarting training: {num_epochs} epochs")
    print(f"Run ID: {run_id}")
    print(f"Monitor endpoint: {monitor_endpoint}")
    print("-" * 60)

    train_acc_history, val_acc_history = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        t_loss, t_corr, n = 0, 0, 0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Log metrics after backward (gradients available) but before optimizer step
            # This captures both gradients (pre-update) and current weights
            monitor.log_metrics()

            optimizer.step()

            # Increment step counter
            monitor.step()

            t_loss += loss.item()
            t_corr += (outputs.argmax(1) == y).sum().item()
            n += len(y)

        # Validation
        model.eval()
        v_corr, m = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                v_corr += (model(x).argmax(1) == y).sum().item()
                m += len(y)

        t_acc, v_acc = t_corr/n, v_corr/m
        train_acc_history.append(t_acc)
        val_acc_history.append(v_acc)

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s): '
              f'Loss: {t_loss/len(train_loader):.4f}, '
              f'Train Acc: {t_acc:.4f}, Val Acc: {v_acc:.4f}')

    print(f'\nFinal Test Accuracy: {v_acc:.4f}')
    print(f'Total training steps: {monitor.global_step}')

    # Close monitor
    monitor.close()

    return model, train_acc_history, val_acc_history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Fashion-MNIST with monitoring')
    parser.add_argument('--model', type=str, default='decent',
                        choices=['minimal', 'decent'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--endpoint', type=str,
                        default='http://localhost:8000/api/v1/metrics/layerwise',
                        help='Monitor server API endpoint')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log metrics every N steps')

    args = parser.parse_args()

    # Select model
    model_class = DecentFashionCNN if args.model == 'decent' else MinimalFashionCNN

    # Train
    train_model(
        model_class=model_class,
        num_epochs=args.epochs,
        monitor_endpoint=args.endpoint,
        log_interval=args.log_interval,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
