import json

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist import DigitCNN, classification_metrics, evaluate_test, predict_image


def test_classification_metrics_counts_true_by_predicted():
    metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1])

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["count"] == 4
    assert metrics["confusion_matrix"][0][0] == 1
    assert metrics["confusion_matrix"][0][1] == 1
    assert metrics["confusion_matrix"][1][1] == 1
    assert metrics["confusion_matrix"][2][1] == 1
    assert sum(map(sum, metrics["confusion_matrix"])) == 4


def test_predict_image_reloads_checkpoint_and_preprocesses(tmp_path):
    model = DigitCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[0].weight[0, 0, 1, 1] = 1
        model.layers[3].weight[0, 0, 1, 1] = 1
        model.layers[7].weight[1, 0] = 4
        model.layers[7].bias[0] = 1
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)

    assert predict_image(checkpoint, Image.new("L", (28, 28), 0), torch.device("cpu")) == 0
    assert predict_image(checkpoint, Image.new("L", (28, 28), 255), torch.device("cpu")) == 1


def test_evaluate_test_saves_report_and_examples(tmp_path):
    class ZeroClassifier(nn.Module):
        def forward(self, images):
            scores = torch.zeros((len(images), 10))
            scores[:, 0] = 1
            return scores

    data = TensorDataset(torch.zeros(3, 1, 28, 28), torch.tensor([0, 1, 2]))
    output = tmp_path / "evaluation"
    report = evaluate_test(
        ZeroClassifier(), DataLoader(data, batch_size=2), torch.device("cpu"), output
    )

    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["confusion_matrix"][1][0] == 1
    assert [(item["index"], item["true"], item["predicted"]) for item in report["examples"]] == [
        (1, 1, 0), (2, 2, 0)
    ]
    assert (output / "report.json").is_file()
    assert all((output / item["image"]).is_file() for item in report["examples"])
    assert json.loads((output / "report.json").read_text()) == report
    with Image.open(output / report["examples"][0]["image"]) as image:
        assert image.size == (28, 28)
        assert image.getpixel((0, 0)) == 0
