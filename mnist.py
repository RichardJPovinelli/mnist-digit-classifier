import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch import nn
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import MNIST


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def fit(
    model: DigitCNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += labels.size(0)
        validation_loss, validation_accuracy = evaluate_epoch(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_count,
                "train_accuracy": train_correct / train_count,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )
    return history


def evaluate_epoch(
    model: DigitCNN, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return loss_total / count, correct / count


def split_indices(size: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def preprocess(image: Image.Image) -> torch.Tensor:
    if image.size != (28, 28):
        raise ValueError("Expected a 28 x 28 image")
    if image.mode != "L":
        raise ValueError("Expected an 8-bit grayscale image")
    return to_tensor(image)


def load_data(
    root: str, batch_size: int, validation_size: int, seed: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = MNIST(root, train=True, download=True, transform=preprocess)
    test = MNIST(root, train=False, download=True, transform=preprocess)
    train_indices, validation_indices = split_indices(len(dataset), validation_size, seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader
