import json
import subprocess
import sys

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import mnist


def test_train_command_records_run_and_refuses_overwrite(monkeypatch, tmp_path):
    images = torch.zeros(8, 1, 28, 28)
    labels = torch.arange(8)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    calls = []

    def fake_load(*args):
        calls.append(args)
        return loader, loader, loader

    monkeypatch.setattr(mnist, "load_data", fake_load)
    run = tmp_path / "run"
    args = [
        "train", "--run-dir", str(run), "--epochs", "1", "--device", "cpu",
        "--batch-size", "4", "--seed", "17"
    ]

    mnist.main(args)

    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())
    assert config["settings"]["epochs"] == 1
    assert config["settings"]["device"] == "cpu"
    assert calls == [(config["settings"]["data_dir"], 4, 5000, 17)]
    assert len(history) == 1
    assert (run / "best.pt").is_file()
    with pytest.raises(FileExistsError):
        mnist.main(args)


def test_evaluate_command_uses_checkpoint_and_writes_report(monkeypatch, tmp_path):
    loader = DataLoader(
        TensorDataset(torch.zeros(8, 1, 28, 28), torch.arange(8)), batch_size=4
    )
    monkeypatch.setattr(mnist, "load_data", lambda *args: (loader, loader, loader))
    run = tmp_path / "run"
    mnist.main(["train", "--run-dir", str(run), "--epochs", "1", "--device", "cpu"])
    model = mnist.DigitCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[7].bias[9] = 1
    torch.save({"model_state": model.state_dict()}, run / "best.pt")

    mnist.main(["evaluate", "--run-dir", str(run), "--device", "cpu"])

    report = json.loads((run / "test-evaluation" / "report.json").read_text())
    assert report["count"] == 8
    assert report["accuracy"] == 0
    assert report["confusion_matrix"][0][9] == 1
    assert sum(map(sum, report["confusion_matrix"])) == 8
    with pytest.raises(FileExistsError):
        mnist.main(["evaluate", "--run-dir", str(run), "--device", "cpu"])


def test_predict_command_prints_digit(tmp_path, capsys):
    model = mnist.DigitCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[7].bias[9] = 1
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    image = tmp_path / "digit.png"
    Image.new("L", (28, 28), 255).save(image)

    mnist.main(["predict", "--checkpoint", str(checkpoint), "--image", str(image), "--device", "cpu"])

    assert capsys.readouterr().out.strip() == "9"


def test_script_exposes_commands():
    result = subprocess.run(
        [sys.executable, mnist.__file__, "--help"],
        check=True, text=True, capture_output=True
    )

    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout


def test_training_records_model_and_runtime_details(monkeypatch, tmp_path):
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 1, 28, 28), torch.arange(4)), batch_size=4
    )
    monkeypatch.setattr(mnist, "load_data", lambda *args: (loader, loader, loader))
    run = tmp_path / "run"

    mnist.main(["train", "--run-dir", str(run), "--epochs", "1", "--device", "cpu"])

    config = json.loads((run / "config.json").read_text())
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model_state"]
    assert config["model"]["parameter_count"] == sum(tensor.numel() for tensor in state.values())
    assert config["model"]["architecture"] == "DigitCNN"
    assert config["packages"]["torch"] == torch.__version__.split("+")[0]
    assert config["torch_threads"] > 0
