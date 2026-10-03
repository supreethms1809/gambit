"""D1–D7 on the val split of CIFAR-10 and HAM10000.

Writes ``results/paper/degenerate/``. Does not touch the test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from base_evidence.gradcam_regions import GradCAMRegionsProvider
from core.device import get_device
from core.game_modes import resolve_contrastive_game, resolve_shift_game
from core.hypotheses import TopMSelector
from core.reporting import save_json
from core.runner import CDEAExplainer
from evaluation.degenerate import (
    D1_MAX_FRACTION_ABOVE_HALF,
    D1_MAX_MASS_FRACTION,
    D2_MAX_SOFT_MINUS_HARD,
    D3_MAX_SUPPRESSION_SHARE,
    D4_MIN_CAPTURE_RATIO,
    D5_MAX_MASS_RATIO,
    D6_MAX_COMPLEMENT_DEVIATION,
    D6_MAX_FRACTION_ABOVE_HALF,
    D6_MIN_COMPLEMENT_MASS,
    d1_shared_blanket,
    d2_soft_versus_hard,
    d3_foil_suppression,
    d4_arbitrary_cells,
    d5_budget,
    d6_shift_masks,
    evidence_capture_ratio,
    same_area_hard,
)
from evaluation.nulls import random_translate
from instantiations.contrastive.allocator import OptimizationAllocator
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.shift.allocator import RobustShortcutOptimizationAllocator
from instantiations.shift.env import default_shift_augs, env_batch_from_augs
from instantiations.shift.objective import RobustShortcutObjective
from modality.grid_regions import VisionGridUnitSpace
from scripts.ablation_contrastive import _build_model, _get_eval_loader

OUT = REPO / "results" / "paper" / "degenerate"
CHECKPOINTS = {
    "cifar10": REPO / "results" / "paper_rerun" / "checkpoints" / "cifar10_resnet18_pt_lp_ep15_lr0.001_seed0.pt",
    "ham10000": REPO / "examples" / "out" / "checkpoints" / "ham10000_resnet18.pt",
}


def _cat(chunks: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(chunks, dim=0)


def _logit(logits: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, classes.unsqueeze(1)).squeeze(1)


def _contrastive(dataset: str, device: torch.device, args: argparse.Namespace) -> dict:
    loader, num_classes = _get_eval_loader(
        dataset, args.batch_size, REPO / "data", image_size=224,
        seed=args.seed, num_images=args.num_images, split="val",
    )
    model = _build_model(
        "resnet18", num_classes, pretrained=False, checkpoint=str(CHECKPOINTS[dataset])
    ).to(device).eval()
    grid = 7
    unit_space = VisionGridUnitSpace(grid, grid)
    game = resolve_contrastive_game("mixed")
    objective = ContrastiveObjective(
        lambda_margin=game.lambda_margin,
        lambda_overlap=game.lambda_overlap,
        lambda_mass=args.lambda_mass,
        lambda_shared_sparse=args.lambda_shared_sparse,
    )
    allocator = OptimizationAllocator(
        objective,
        num_steps=args.num_steps,
        lr=0.2,
        use_shared=game.use_shared,
        lambda_partition=game.lambda_partition,
    )
    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=TopMSelector(m=min(5, num_classes)),
        base_evidence=GradCAMRegionsProvider(grid, grid),
        allocator=allocator,
        objective=objective,
        normalize_evidence=True,
        device=device,
    )

    shared, unique, evidence, z_soft, z_hard = [], [], [], [], []
    z_k_keep, z_l_keep, z_k_full, z_l_full = [], [], [], []
    correct = total = 0
    seen = 0
    for x, y in loader:
        if seen >= args.num_images:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            full = model(x)
        correct += int((full.argmax(1) == y).sum().item())
        total += int(y.numel())
        explanation = explainer.explain(x)
        masks = explanation.masks
        hypotheses = explanation.hypotheses
        k = hypotheses.ids[:, 0]
        foil = hypotheses.ids[:, 1]
        m_shared = masks["shared"]
        m_top = (masks["unique"][:, 0] + m_shared).clamp(0, 1)
        hard = same_area_hard(m_top)
        with torch.no_grad():
            kept = model(unit_space.keep(x, m_top))
            kept_hard = model(unit_space.keep(x, hard))
        shared.append(m_shared.detach().cpu())
        unique.append(masks["unique"].detach().cpu())
        evidence.append(explanation.extras["evidence"][:, 0].detach().cpu())
        z_soft.append(_logit(kept, k).detach().cpu())
        z_hard.append(_logit(kept_hard, k).detach().cpu())
        z_k_keep.append(_logit(kept, k).detach().cpu())
        z_l_keep.append(_logit(kept, foil).detach().cpu())
        z_k_full.append(_logit(full, k).detach().cpu())
        z_l_full.append(_logit(full, foil).detach().cpu())
        seen += x.shape[0]
        print(f"  contrastive {dataset}: {seen}/{args.num_images}", flush=True)

    shared_t = _cat(shared)
    unique_t = _cat(unique)
    evidence_t = _cat(evidence)
    top_unique = unique_t[:, 0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    translated = random_translate(top_unique, generator, grid, grid)
    regions = top_unique.shape[-1]
    mass_scale = max(1.0, regions / float(objective.mass_ref_regions))
    # Evidence is normalised to sum to one per hypothesis, which is the objective's target before scaling.
    target = torch.full((unique_t.shape[0], unique_t.shape[1]), mass_scale)
    d1 = d1_shared_blanket(shared_t)
    d1["capture_ratio"] = float(evidence_capture_ratio(shared_t, evidence_t).mean().item())
    return {
        "n": int(seen),
        "val_accuracy": correct / max(total, 1),
        "lambda_shared_sparse": objective.lambda_shared_sparse,
        "lambda_mass": objective.lambda_mass,
        "lambda_overlap": objective.lambda_overlap,
        "num_steps": args.num_steps,
        "D1": d1,
        "D2": d2_soft_versus_hard(_cat(z_soft), _cat(z_hard)),
        "D3": d3_foil_suppression(_cat(z_k_keep), _cat(z_l_keep), _cat(z_k_full), _cat(z_l_full)),
        "D4": d4_arbitrary_cells(
            evidence_capture_ratio(top_unique, evidence_t),
            evidence_capture_ratio(translated, evidence_t),
        ),
        "D5": d5_budget(unique_t.sum(dim=-1), target),
    }


def _shift(dataset: str, device: torch.device, args: argparse.Namespace, model: torch.nn.Module) -> dict:
    loader, num_classes = _get_eval_loader(
        dataset, args.batch_size, REPO / "data", image_size=224,
        seed=args.seed, num_images=args.num_images, split="val",
    )
    grid = 7
    unit_space = VisionGridUnitSpace(grid, grid)
    game = resolve_shift_game("mixed")
    objective = RobustShortcutObjective(
        lambda_mean=game.lambda_mean,
        lambda_var=game.lambda_var,
        lambda_gap=game.lambda_gap,
        lambda_shortcut=game.lambda_shortcut,
        lambda_disjoint=game.lambda_disjoint,
        lambda_sparse=game.lambda_sparse,
        lambda_mass=args.lambda_mass,
    )
    allocator = RobustShortcutOptimizationAllocator(
        objective, num_steps=args.num_steps, lr=0.2, lambda_disjoint=game.lambda_disjoint,
    )
    provider = GradCAMRegionsProvider(grid, grid)
    selector = TopMSelector(m=min(5, num_classes))
    aug1, aug2 = default_shift_augs()
    robust, shortcut = [], []
    seen = 0
    for x, _y in loader:
        if seen >= args.num_images:
            break
        x = x.to(device)
        env = env_batch_from_augs(x, aug1, aug2)
        with torch.no_grad():
            logits = model(x)
            hypotheses = selector.select(logits, torch.softmax(logits, dim=-1))
        evidence = provider.explain(x.detach().clone().requires_grad_(True), model, hypotheses)
        evidence = evidence.detach() / evidence.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        masks = allocator.allocate(
            x=x, model=model, unit_space=unit_space, hypotheses=hypotheses,
            evidence=evidence, env=env,
        )
        robust.append(masks["robust"].detach().cpu())
        shortcut.append(masks["shortcut"].detach().cpu())
        seen += x.shape[0]
        print(f"  shift {dataset}: {seen}/{args.num_images}", flush=True)
    return {
        "n": int(seen),
        "lambda_mass": objective.lambda_mass,
        "lambda_disjoint": objective.ld,
        "num_steps": args.num_steps,
        "D6": d6_shift_masks(_cat(robust), _cat(shortcut)),
    }


def _status(flag: bool) -> str:
    return "open" if flag else "closed"


def _report(payload: dict) -> str:
    lines = [
        "# Degenerate-optimum checks",
        "",
        "Val split only. Seeded subset. The thresholds are the constraints named in `evaluation/degenerate.py`.",
        "",
        f"- D1 closed when the shared-mask fraction above 0.5 is at most {D1_MAX_FRACTION_ABOVE_HALF} "
        f"and the mass fraction is at most {D1_MAX_MASS_FRACTION}.",
        f"- D2 closed when the soft kept logit exceeds the same-area hard mask by at most {D2_MAX_SOFT_MINUS_HARD}.",
        f"- D3 closed when the share of a risen margin coming from foil suppression is at most {D3_MAX_SUPPRESSION_SHARE}.",
        f"- D4 closed when unique-mask evidence capture is at least {D4_MIN_CAPTURE_RATIO} times chance and above a translated copy.",
        f"- D5 closed when unique-mask mass over the mass target is at most {D5_MAX_MASS_RATIO}.",
        f"- D6 closed when the robust-mask fraction above 0.5 is at most {D6_MAX_FRACTION_ABOVE_HALF}, "
        f"and the shortcut mask is not a copy of the robust complement "
        f"(mean absolute deviation at least {D6_MAX_COMPLEMENT_DEVIATION}, "
        f"or shortcut mass fraction at most {D6_MIN_COMPLEMENT_MASS}).",
        "- D7 closed by recording the ID-OOD gap as a model property on the full image, "
        "in `scripts/eval_robust_shortcut.py` and `scripts/run_experiments.py`.",
        "",
    ]
    for name, block in payload["datasets"].items():
        c = block["contrastive"]
        s = block["shift"]
        lines += [
            f"## {name}",
            "",
            f"Checkpoint `{block['checkpoint']}`. n={c['n']}. "
            f"Val accuracy on this subset: {c['val_accuracy']:.3f}. "
            f"Allocator steps: {c['num_steps']}.",
            "",
            "| Route | Status | Measurement |",
            "|---|---|---|",
            f"| D1 shared blanket | {_status(c['D1']['open'])} | "
            f"fraction above 0.5 = {c['D1']['fraction_above_half']:.4f}, "
            f"mass fraction = {c['D1']['mass_fraction']:.4f}, "
            f"evidence capture / area = {c['D1']['capture_ratio']:.4f} |",
            f"| D2 soft vs hard | {_status(c['D2']['open'])} | "
            f"soft minus hard logit = {c['D2']['soft_minus_hard']:.4f} |",
            f"| D3 foil suppression | {_status(c['D3']['open'])} | "
            f"suppression share = {c['D3']['suppression_share']:.4f}, "
            f"z_k keep/full = {c['D3']['z_k_keep']:.3f}/{c['D3']['z_k_full']:.3f}, "
            f"z_l keep/full = {c['D3']['z_l_keep']:.3f}/{c['D3']['z_l_full']:.3f} |",
            f"| D4 arbitrary cells | {_status(c['D4']['open'])} | "
            f"capture ratio = {c['D4']['capture_ratio']:.4f}, "
            f"translated null = {c['D4']['null_capture_ratio']:.4f} |",
            f"| D5 budget | {_status(c['D5']['open'])} | mass ratio = {c['D5']['mass_ratio']:.4f} |",
            f"| D6 shift masks | {_status(s['D6']['open'])} | "
            f"robust fraction above 0.5 = {s['D6']['robust_fraction_above_half']:.4f}, "
            f"complement deviation = {s['D6']['complement_deviation']:.4f}, "
            f"shortcut mass fraction = {s['D6']['shortcut_mass_fraction']:.4f} |",
            f"| D7 model gap | closed | moved out of the method table |",
            "",
        ]
    open_routes = [
        f"{name} {route}"
        for name, block in payload["datasets"].items()
        for route, section in (
            ("D1", block["contrastive"]["D1"]),
            ("D2", block["contrastive"]["D2"]),
            ("D3", block["contrastive"]["D3"]),
            ("D4", block["contrastive"]["D4"]),
            ("D5", block["contrastive"]["D5"]),
            ("D6", block["shift"]["D6"]),
        )
        if section["open"]
    ]
    lines.append("Open routes: " + (", ".join(open_routes) if open_routes else "none") + ".")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Degenerate-optimum checks on val")
    parser.add_argument("--num-images", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "ham10000"])
    parser.add_argument("--lambda-mass", type=float, default=0.1)
    parser.add_argument("--lambda-shared-sparse", type=float, default=0.25)
    parser.add_argument("--skip-shift", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(OUT))
    args = parser.parse_args()
    if args.num_images <= 0 or args.num_steps <= 0:
        raise SystemExit("num-images and num-steps must be positive")
    device = get_device()
    print("Device:", device)
    payload = {
        "split": "val",
        "seed": args.seed,
        "device": str(device),
        "datasets": {},
    }
    for dataset in args.datasets:
        checkpoint = CHECKPOINTS[dataset]
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        print(f"=== {dataset} ===", flush=True)
        contrastive = _contrastive(dataset, device, args)
        model = _build_model(
            "resnet18",
            10 if dataset == "cifar10" else 7,
            pretrained=False,
            checkpoint=str(checkpoint),
        ).to(device).eval()
        shift = None if args.skip_shift else _shift(dataset, device, args, model)
        payload["datasets"][dataset] = {
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "contrastive": contrastive,
            "shift": shift,
        }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["lambda_mass"] = args.lambda_mass
    payload["lambda_shared_sparse"] = args.lambda_shared_sparse
    save_json(out_dir / "checks.json", payload, config_hash=None, device=str(device))
    if args.skip_shift:
        for name, block in payload["datasets"].items():
            c = block["contrastive"]
            print(
                f"{name} mass={args.lambda_mass} shared={args.lambda_shared_sparse} "
                f"D1={c['D1']['mass_fraction']:.3f}/{c['D1']['fraction_above_half']:.3f} "
                f"cap={c['D1']['capture_ratio']:.3f} "
                f"D2={c['D2']['soft_minus_hard']:.3f} "
                f"D3={c['D3']['suppression_share']:.3f} "
                f"D4={c['D4']['capture_ratio']:.3f} "
                f"D5={c['D5']['mass_ratio']:.3f} "
                f"zk={c['D3']['z_k_keep']:.2f}"
            )
        return
    report = _report(payload)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
