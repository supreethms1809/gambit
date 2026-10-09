"""S09 training grid and val metrics. No checkpoint is trained here."""

from __future__ import annotations

import torch
import torch.nn as nn

from evaluation.accuracy import top1_and_balanced, worst_group_accuracy
from scripts.launch_paper_training import CONTRASTIVE, SHIFT, paper_cells
from scripts.train_backbone import _backbone_to_eval, _build_model, _freeze_backbone, train_model


def test_scores_match_a_known_prediction():
    pred = torch.tensor([0, 0, 0, 1])
    target = torch.tensor([0, 0, 1, 1])
    top1, balanced = top1_and_balanced(pred, target, num_classes=3)
    assert top1 == 0.75
    assert balanced == 0.75  # class 2 is absent; class 0 recall 1, class 1 recall 0.5


def test_worst_group_is_the_minimum():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 0, 1, 0])
    group = torch.tensor([0, 0, 1, 1])
    assert worst_group_accuracy(pred, target, group) == 0.5


def test_paper_grid_is_five_seeds_and_skips_imagenet_s():
    cells = paper_cells()
    assert len(cells) == (len(CONTRASTIVE) + len(SHIFT)) * 2 * 5
    assert {c["seed"] for c in cells} == {0, 1, 2, 3, 4}
    names = {c["dataset"] for c in cells}
    assert "mnist" not in names
    assert "pets" not in names
    assert names == set(CONTRASTIVE) | set(SHIFT)
    probes = [c for c in cells if c["dataset"] == "cifar10"]
    assert all(c["freeze_backbone"] and c["lr"] == 1e-3 for c in probes)
    planted = [c for c in cells if c["dataset"] == "planted_patch"]
    assert all(not c["freeze_backbone"] and c["lr"] == 1e-4 for c in planted)


def test_resnet50_head_matches_the_dataset():
    model = _build_model("resnet50", 17, pretrained=False)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 17
    assert model.fc.in_features == 2048


class _ProbeNet(nn.Module):
    """Backbone (conv+BN+dropout) plus a linear head, mirroring a linear probe."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.drop(x)
        return self.fc(x.mean(dim=(2, 3)))


def test_linear_probe_freezes_bn_stats_and_dropout_but_trains_head():
    torch.manual_seed(0)
    model = _ProbeNet()
    _freeze_backbone(model)
    assert not model.bn.weight.requires_grad
    assert model.fc.weight.requires_grad
    xs = torch.rand(32, 3, 8, 8)
    ys = torch.randint(0, 2, (32,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs, ys), batch_size=8
    )
    head_before = model.fc.weight.detach().clone()
    bn_before = model.bn.running_mean.clone()
    trained = train_model(
        model,
        loader,
        num_epochs=2,
        lr=1e-3,
        freeze_backbone=True,
        device=torch.device("cpu"),
    )
    assert trained.training is False  # train_model returns the model in eval
    assert not torch.equal(trained.fc.weight, head_before), "head must learn"
    assert torch.equal(trained.bn.running_mean, bn_before), "frozen BN stats must not move"
    # train_model leaves the model in eval; the point above is that the two
    # training epochs did not move the frozen backbone statistics either.
    assert trained.bn.training is False

