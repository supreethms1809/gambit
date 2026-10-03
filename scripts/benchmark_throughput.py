"""Forward/backward throughput on MPS, used to set per-dataset n.

Random initialisation is enough: the weights do not change the flop count, and
this script does not download them. Pass ``--allow-download`` only after that
download has been approved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def benchmark(model: nn.Module, batch: torch.Tensor, steps: int) -> dict:
    device = batch.device
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(batch).sum().backward()
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(batch).sum().backward()
    _sync(device)
    elapsed = time.perf_counter() - start
    return {
        "steps": steps,
        "batch_size": int(batch.shape[0]),
        "seconds": elapsed,
        "iterations_per_second": steps / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure forward/backward throughput")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out", type=str, default="results/paper/logs/throughput.json")
    args = parser.parse_args()

    from torchvision import models

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    weights = "DEFAULT" if args.allow_download else None
    builders = {
        "resnet50": lambda: models.resnet50(weights=weights),
        "vit_b_16": lambda: models.vit_b_16(weights=weights),
    }
    batch = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
    report = {"device": str(device), "weights": "default" if weights else "random", "models": {}}
    for name, build in builders.items():
        print(f"benchmarking {name} on {device}")
        stats = benchmark(build(), batch, args.steps)
        report["models"][name] = stats
        print(f"  {stats['iterations_per_second']:.2f} forward+backward/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
