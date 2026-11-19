import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import mnist
from mnist import DigitCNN, fit


def test_cnn_produces_trainable_logits():
    torch.manual_seed(7)
    model = DigitCNN()
    logits = model(torch.rand(3, 1, 28, 28))
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 4, 9]))
    loss.backward()

    assert logits.shape == (3, 10)
    assert not torch.equal(logits[0], logits[1])
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_fit_trains_and_saves_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = DigitCNN()
    before = model.layers[0].weight.detach().clone()
    data = TensorDataset(torch.rand(8, 1, 28, 28), torch.arange(8))
    loader = DataLoader(data, batch_size=4)
    checkpoint = tmp_path / "best.pt"

    history = fit(model, loader, loader, 1, 0.001, checkpoint, torch.device("cpu"))

    assert len(history) == 1
    assert not torch.equal(before, model.layers[0].weight)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 1
    assert torch.equal(saved["model_state"]["layers.0.weight"], model.layers[0].weight)


def test_fit_keeps_best_validation_checkpoint(monkeypatch, tmp_path):
    results = iter([(1.1, 0.7), (1.0, 0.8), (0.9, 0.6)])
    monkeypatch.setattr(mnist, "evaluate_epoch", lambda *_: next(results))
    data = TensorDataset(torch.rand(8, 1, 28, 28), torch.arange(8))
    loader = DataLoader(data, batch_size=4)
    checkpoint = tmp_path / "best.pt"

    history = fit(DigitCNN(), loader, loader, 3, 0.001, checkpoint, torch.device("cpu"))

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert [row["validation_accuracy"] for row in history] == [0.7, 0.8, 0.6]
    assert saved["epoch"] == 2
    assert saved["validation_accuracy"] == 0.8
