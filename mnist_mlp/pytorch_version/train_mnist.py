import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTClassifier(nn.Module):
    """A simple fully connected network for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on the evaluation dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main() -> None:
    # Use CUDA if available, otherwise train on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert images to tensors and normalize pixel values to a stable range.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Download MNIST and build data loaders for training and testing.
    # Both this PyTorch example and the C++ implementation share the repo-level
    # data/MNIST/raw directory. torchvision appends MNIST/raw under root.
    data_root = Path(__file__).resolve().parent.parent / "data"
    train_dataset = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the model, loss function, and optimizer.
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Standard training loop over a small number of epochs.
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"Training Loss: {average_loss:.4f} - "
            f"Test Accuracy: {accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
