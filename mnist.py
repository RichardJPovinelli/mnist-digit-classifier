import argparse
import json
import math
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_pil_image, to_tensor


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


def classification_metrics(labels: list[int], predictions: list[int]) -> dict:
    matrix = [[0] * 10 for _ in range(10)]
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[label][prediction] += 1
    return {
        "count": len(labels),
        "accuracy": sum(matrix[i][i] for i in range(10)) / len(labels),
        "confusion_matrix": matrix,
    }


def predict_image(checkpoint_path: Path, image: Image.Image, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        return int(model(preprocess(image).unsqueeze(0).to(device)).argmax(dim=1).item())


def evaluate_test(
    model: nn.Module, loader: DataLoader, device: torch.device, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=False, exist_ok=False)
    model.to(device).eval()
    labels_all: list[int] = []
    predictions_all: list[int] = []
    examples = []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()
            label_values = labels.tolist()
            for offset, (label, prediction) in enumerate(zip(label_values, predictions, strict=True)):
                if label != prediction and len(examples) < 10:
                    index = len(labels_all) + offset
                    filename = f"misclassified-{index}.png"
                    to_pil_image(images[offset]).save(output_dir / filename)
                    examples.append(
                        {"index": index, "true": label, "predicted": prediction, "image": filename}
                    )
            labels_all.extend(label_values)
            predictions_all.extend(predictions)
    report = classification_metrics(labels_all, predictions_all)
    report["examples"] = examples
    with (output_dir / "report.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    run_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    validation_size: int = 5000

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be between 0 and 2**63 - 1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and use an MNIST digit classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.command == "predict":
        with Image.open(args.image) as image:
            print(predict_image(args.checkpoint, image, torch.device(device_name)))
        return
    if args.command == "evaluate":
        run_dir = args.run_dir.resolve()
        output_dir = run_dir / "test-evaluation"
        if output_dir.exists():
            raise FileExistsError(f"Test evaluation already exists: {output_dir}")
        with (run_dir / "config.json").open(encoding="utf-8") as file:
            config = RunConfig(**json.load(file)["settings"])
        device = torch.device(device_name)
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model = DigitCNN().to(device)
        model.load_state_dict(checkpoint["model_state"])
        _, _, test_loader = load_data(
            config.data_dir, config.batch_size, config.validation_size, config.seed
        )
        report = evaluate_test(model, test_loader, device, output_dir)
        print(json.dumps(report, indent=2))
        return
    config = RunConfig(
        data_dir=str(args.data_dir.resolve()),
        run_dir=str(args.run_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device_name,
    )
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = DigitCNN()
    metadata = {
        "settings": asdict(config),
        "dataset": "torchvision MNIST: 60000 official training, 10000 official test",
        "split": "seeded torch.randperm of official training set; first 5000 for validation",
        "preprocessing": "8-bit grayscale pixels divided by 255",
        "model": {
            "architecture": type(model).__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "optimizer": "Adam",
        "loss": "cross entropy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in distributions()
            if dist.metadata.get("Name")
        },
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "artifacts": {"checkpoint": "best.pt", "history": "history.json", "test": "test-evaluation"},
    }
    with (run_dir / "config.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    train_loader, validation_loader, _ = load_data(
        config.data_dir, config.batch_size, config.validation_size, config.seed
    )
    history = fit(
        model, train_loader, validation_loader, config.epochs,
        config.learning_rate, run_dir / "best.pt", device
    )
    with (run_dir / "history.json").open("x", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    for row in history:
        print(json.dumps(row))


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


if __name__ == "__main__":
    main()
