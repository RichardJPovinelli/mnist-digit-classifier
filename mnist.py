import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import MNIST


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
