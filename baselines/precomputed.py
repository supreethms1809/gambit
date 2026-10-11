"""Load allocation maps produced outside this tree.

A8 compares against maps dumped from the earlier formulation. This module only
reads them. The dump script does not live here.
"""

from __future__ import annotations

from pathlib import Path

import torch

from core.types import HypothesisSet


def load_pair(directory: str | None, hypotheses: HypothesisSet, grid: tuple[int, int], *, index: list[int] | None = None, area: float | None = None):
    """``.npz`` maps keyed by image index. The stored class ids must match ``hypotheses``."""
    from evaluation.run_methods import PairMaps

    if not directory:
        raise ValueError("precomputed maps need a directory")
    path = Path(directory)
    if path.is_dir():
        tag = f"a{area:.4f}" if area is not None else "maps"
        files = sorted(path.glob(f"*{tag}*.npz")) or sorted(path.glob("*.npz"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one npz in {path}, found {len(files)}")
        path = files[0]
    blob = dict(torch.load(path, map_location="cpu", weights_only=False)) if path.suffix == ".pt" else None
    if blob is None:
        import numpy as np

        loaded = np.load(path)
        blob = {key: torch.as_tensor(loaded[key]) for key in loaded.files}
    stored_k = blob["class_k"].long()
    stored_l = blob["class_l"].long()
    if index is not None:
        positions = []
        stored_index = blob["index"].long().tolist()
        for item in index:
            if int(item) not in stored_index:
                raise KeyError(f"image index {item} is not in {path}")
            positions.append(stored_index.index(int(item)))
        selector = torch.tensor(positions)
        stored_k = stored_k[selector]
        stored_l = stored_l[selector]
        k_map = blob["k"][selector]
        l_map = blob["l"][selector]
    else:
        k_map = blob["k"]
        l_map = blob["l"]
    if not torch.equal(stored_k, hypotheses.ids[:, 0].detach().cpu().long()):
        raise ValueError("precomputed class k does not match this sample")
    if not torch.equal(stored_l, hypotheses.ids[:, 1].detach().cpu().long()):
        raise ValueError("precomputed class l does not match this sample")
    extra = {}
    if "shared" in blob:
        extra["shared"] = blob["shared"] if index is None else blob["shared"][selector]
    return PairMaps(k=k_map.to(hypotheses.ids.device), l=l_map.to(hypotheses.ids.device), grid=grid, extra=extra)
