"""Raw-input wrapper, provenance on every JSON, and the dirty-tree guard."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from core.reporting import config_hash, refuse_final_if_dirty, save_json
from models.wrapper import IMAGENET_MEAN, IMAGENET_STD, NormalizedModel


class _Capture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last = x.detach().clone()
        return self.conv(x)


def test_wrapper_normalises_raw_input_and_rejects_standardised_input():
    inner = _Capture()
    wrapped = NormalizedModel(inner)
    raw = torch.rand(2, 3, 8, 8)
    out = wrapped(raw)
    assert out.shape == (2, 4, 8, 8)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert torch.allclose(inner.last, (raw - mean) / std)
    with pytest.raises(ValueError, match="raw"):
        wrapped(raw * 2)


def test_wrapper_forwards_inner_layers():
    wrapped = NormalizedModel(_Capture())
    assert isinstance(wrapped.conv, nn.Conv2d)


def test_save_json_records_provenance(tmp_path):
    path = tmp_path / "out.json"
    digest = config_hash({"lr": 0.2, "steps": 40})
    save_json(path, {"metric": 1.0}, config_hash=digest, device="cpu", data_split_hash="abc")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["config_hash"] == digest
    assert provenance["data_split_hash"] == "abc"
    assert provenance["device"] == "cpu"
    assert provenance["torch"]
    assert "git_commit" in provenance
    assert isinstance(provenance["git_dirty"], bool)
    assert provenance["timestamp"]
    assert "nan" not in path.read_text(encoding="utf-8").lower() or payload["metric"] == 1.0


def test_save_json_maps_non_finite_to_null(tmp_path):
    path = tmp_path / "nan.json"
    save_json(path, {"value": float("nan")}, provenance=False)
    assert json.loads(path.read_text(encoding="utf-8"))["value"] is None


def test_refuse_final_on_a_dirty_tree(monkeypatch):
    monkeypatch.setattr("core.reporting.git_state", lambda: ("abc123", True))
    with pytest.raises(RuntimeError, match="dirty"):
        refuse_final_if_dirty(True)
    refuse_final_if_dirty(False)

    monkeypatch.setattr("core.reporting.git_state", lambda: ("abc123", False))
    refuse_final_if_dirty(True)
