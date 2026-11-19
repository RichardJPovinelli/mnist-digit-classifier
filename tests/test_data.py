import torch
import pytest
from PIL import Image

import mnist
from mnist import load_data, preprocess, split_indices


def test_split_indices_is_reproducible_and_disjoint():
    train, validation = split_indices(10, validation_size=3, seed=7)
    repeat_train, repeat_validation = split_indices(10, validation_size=3, seed=7)

    assert train == repeat_train
    assert validation == repeat_validation
    assert len(train) == 7
    assert len(validation) == 3
    assert set(train).isdisjoint(validation)
    assert set(train + validation) == set(range(10))


def test_split_indices_uses_seed():
    _, first = split_indices(100, validation_size=20, seed=7)
    _, second = split_indices(100, validation_size=20, seed=8)

    assert first != second


def test_preprocess_scales_grayscale_pixels():
    image = Image.new("L", (28, 28), color=255)
    image.putpixel((1, 0), 0)
    image.putpixel((2, 0), 128)

    pixels = preprocess(image)

    assert pixels.shape == (1, 28, 28)
    assert pixels[0, 0, 0].item() == 1.0
    assert pixels[0, 0, 1].item() == 0.0
    assert pixels[0, 0, 2].item() == pytest.approx(128 / 255)
    assert pixels.dtype == torch.float32


def test_preprocess_rejects_non_mnist_dimensions():
    with pytest.raises(ValueError, match="28 x 28"):
        preprocess(Image.new("L", (20, 28)))


def test_preprocess_rejects_color_images():
    with pytest.raises(ValueError, match="grayscale"):
        preprocess(Image.new("RGB", (28, 28)))


def test_load_data_builds_disjoint_batches(monkeypatch, tmp_path):
    calls = []

    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            calls.append((root, train, download))
            self.size = 10 if train else 4
            self.transform = transform

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return self.transform(Image.new("L", (28, 28), index)), index

    monkeypatch.setattr(mnist, "MNIST", FakeMNIST)
    train, validation, test = load_data(str(tmp_path), 3, 2, seed=7)
    train_labels = {label.item() for _, labels in train for label in labels}
    validation_labels = {label.item() for _, labels in validation for label in labels}

    assert train_labels.isdisjoint(validation_labels)
    assert train_labels | validation_labels == set(range(10))
    assert len(test.dataset) == 4
    assert calls == [(str(tmp_path), True, True), (str(tmp_path), False, True)]
    images, _ = next(iter(test))
    assert images.shape == (3, 1, 28, 28)


def test_load_data_surfaces_download_errors(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(mnist, "MNIST", fail_download)
    with pytest.raises(RuntimeError, match="download failed"):
        load_data(str(tmp_path), 3, 2, seed=7)
