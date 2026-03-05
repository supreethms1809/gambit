"""
Contrastive explanation example using Integrated Gradients as base evidence.

This keeps the same CDEA pipeline as examples/contrastive_explanation.py, but swaps:
  - Evidence provider: IntegratedGradientsRegionsProvider instead of GradCAMRegionsProvider.

Run from repo root:
  PYTHONPATH=. python examples/contrastive_explanation_ig.py --dataset mnist
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.device import get_device
from core.interaction import get_interaction
from core.game_modes import contrastive_game_modes, resolve_contrastive_game
from modality.grid_regions import VisionGridUnitSpace
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from examples.contrastive_explanation import (
    TV_INPUT_SIZE,
    _get_dataloaders,
    _load_batch,
    _load_batch_with_labels,
    get_torchvision_model,
    load_checkpoint,
    save_contrastive_reports,
    save_checkpoint,
    train_model,
    visualize_contrastive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive explanation example (Integrated Gradients evidence)")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "pets", "stanford_dogs"], default="cifar10")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_viz_classes", type=int, default=3,
                        help="Maximum top classes to display in the figure (reduces visual clutter)")
    parser.add_argument("--num_alloc_steps", type=int, default=25)
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--interaction", type=str, default="none", choices=["none", "attention", "transformer"])
    parser.add_argument("--interaction_dim", type=int, default=64)
    parser.add_argument("--interaction_heads", type=int, default=4)
    parser.add_argument("--interaction_ffn", type=int, default=256)
    parser.add_argument("--interaction_dropout", type=float, default=0.0)
    parser.add_argument(
        "--game_mode",
        type=str,
        choices=list(contrastive_game_modes()),
        default="mixed",
        help="Contrastive game mode: cooperative, competitive, mixed, or manual (explicit lambdas)",
    )
    parser.add_argument(
        "--interaction_attn_mix",
        type=float,
        default=0.35,
        help="Blend ratio in [0,1] for attention-conditioned evidence init in allocator",
    )
    parser.add_argument(
        "--interaction_weight_blend",
        type=float,
        default=0.5,
        help="Blend ratio in [0,1] for attention-conditioned hypothesis weighting in objective",
    )
    parser.add_argument("--use_shared", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_partition", type=float, default=0.1)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--lambda_overlap", type=float, default=0.2)
    parser.add_argument("--lambda_disjoint", type=float, default=0.1)
    parser.add_argument("--lambda_mass", type=float, default=2.0,
                        help="Penalty weight to keep optimized mask mass close to base evidence mass")
    parser.add_argument("--ig_steps", type=int, default=24, help="Integrated Gradients interpolation steps")
    parser.add_argument("--ig_baseline", type=str, default="zero", choices=["zero", "mean"],
                        help="Baseline for Integrated Gradients path integral")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.interaction != "none":
        if args.interaction_dim <= 0:
            parser.error("--interaction_dim must be > 0 when --interaction is enabled")
        if args.interaction_heads <= 0:
            parser.error("--interaction_heads must be > 0 when --interaction is enabled")
        if args.interaction_dim % args.interaction_heads != 0:
            parser.error("--interaction_dim must be divisible by --interaction_heads")
        if args.interaction == "transformer" and args.interaction_ffn <= 0:
            parser.error("--interaction_ffn must be > 0 for --interaction transformer")
    if args.lambda_partition < 0:
        parser.error("--lambda_partition must be >= 0")
    if args.lambda_margin < 0:
        parser.error("--lambda_margin must be >= 0")
    if args.lambda_overlap < 0:
        parser.error("--lambda_overlap must be >= 0")
    if args.lambda_disjoint < 0:
        parser.error("--lambda_disjoint must be >= 0")
    if args.lambda_mass < 0:
        parser.error("--lambda_mass must be >= 0")
    if args.max_viz_classes <= 0:
        parser.error("--max_viz_classes must be > 0")
    if not (0.0 <= args.interaction_attn_mix <= 1.0):
        parser.error("--interaction_attn_mix must be in [0, 1]")
    if not (0.0 <= args.interaction_weight_blend <= 1.0):
        parser.error("--interaction_weight_blend must be in [0, 1]")
    if args.ig_steps <= 0:
        parser.error("--ig_steps must be > 0")

    manual_flags = {"--use_shared", "--no-use_shared", "--lambda_margin", "--lambda_partition", "--lambda_overlap", "--lambda_disjoint"}
    used_manual_flags = sorted(f for f in manual_flags if f in sys.argv[1:])
    effective_game_mode = args.game_mode
    if args.game_mode != "manual" and used_manual_flags:
        effective_game_mode = "manual"
        print(
            "Detected explicit shared/lambda flags (%s); switching game_mode to manual."
            % ", ".join(used_manual_flags)
        )
    if effective_game_mode == "manual":
        game_cfg = resolve_contrastive_game(
            "manual",
            use_shared=bool(args.use_shared),
            lambda_margin=float(args.lambda_margin),
            lambda_overlap=float(args.lambda_overlap),
            lambda_disjoint=float(args.lambda_disjoint),
            lambda_partition=float(args.lambda_partition),
        )
    else:
        game_cfg = resolve_contrastive_game(effective_game_mode)

    device = get_device()
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Model:", args.model, "(pretrained)" if args.pretrained else "")
    print("Evidence: integrated_gradients (steps=%d baseline=%s)" % (args.ig_steps, args.ig_baseline))
    if args.interaction == "none":
        print("Interaction: none")
    else:
        print(
            "Interaction: %s (dim=%d heads=%d ffn=%d dropout=%.3f attn_mix=%.2f weight_blend=%.2f)"
            % (
                args.interaction,
                args.interaction_dim,
                args.interaction_heads,
                args.interaction_ffn,
                args.interaction_dropout,
                args.interaction_attn_mix,
                args.interaction_weight_blend,
            )
        )
    print(
        "Game mode: %s (shared=%s margin=%.3f overlap=%.3f disjoint=%.3f partition=%.3f)"
        % (
            game_cfg.mode,
            "on" if game_cfg.use_shared else "off",
            game_cfg.lambda_margin,
            game_cfg.lambda_overlap,
            game_cfg.lambda_disjoint,
            game_cfg.lambda_partition,
        )
    )

    if args.checkpoint is None and args.train:
        args.checkpoint = str(REPO / "examples" / "out" / "checkpoints" / f"{args.dataset}_{args.model}.pt")
    elif args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)

    model = None
    class_names = None
    num_classes = None
    dataset_for_batch = args.dataset

    if args.train:
        print("Training...")
        train_loader, val_loader, class_names, num_classes = _get_dataloaders(args.dataset, args.batch_size, tv_size=TV_INPUT_SIZE)
        model = get_torchvision_model(args.model, num_classes, pretrained=args.pretrained)
        model, best_acc = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
        print("Best val accuracy: %.4f" % best_acc)
        save_checkpoint(Path(args.checkpoint), model, args.dataset, args.model, num_classes)
    elif args.checkpoint is not None:
        model, dataset_for_batch, class_names, num_classes = load_checkpoint(args.checkpoint, device)
        print("Loaded model from", args.checkpoint)
    else:
        try:
            _, class_names, num_classes = _load_batch(args.dataset, args.batch_size, device)
        except Exception:
            if args.dataset == "pets":
                num_classes = 2
            elif args.dataset == "stanford_dogs":
                num_classes = 120
            else:
                num_classes = 10
            class_names = [str(i) for i in range(num_classes)]
        model = get_torchvision_model(args.model, num_classes, pretrained=args.pretrained)

    if model is None:
        raise RuntimeError("Model not set")
    model = model.to(device).eval()
    if class_names is None or num_classes is None:
        _, class_names, num_classes = _load_batch(dataset_for_batch, args.batch_size, device)

    try:
        x, y_true, class_names, num_classes = _load_batch_with_labels(dataset_for_batch, args.batch_size, device)
    except FileNotFoundError as e:
        print("Dataset not found in data/. Using random batch. Error:", e)
        if dataset_for_batch == "mnist":
            hw = 28
            num_classes = 10
        elif dataset_for_batch == "cifar10":
            hw = 32
            num_classes = 10
        elif dataset_for_batch == "pets":
            hw = 64
            num_classes = 2
        else:
            hw = TV_INPUT_SIZE
            num_classes = 120
        x = torch.rand(args.batch_size, 3, hw, hw, device=device)
        y_true = torch.randint(0, num_classes, (args.batch_size,), device=device)
        class_names = [str(i) for i in range(num_classes)]

    if x.shape[2] != TV_INPUT_SIZE or x.shape[3] != TV_INPUT_SIZE:
        x = F.interpolate(x, size=(TV_INPUT_SIZE, TV_INPUT_SIZE), mode="bilinear", align_corners=False)

    top_k = min(5, num_classes)
    grid_h, grid_w = 7, 7

    embed_dim = args.interaction_dim if args.interaction != "none" else None
    unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur", embed_dim=embed_dim)
    selector = TopMSelector(m=top_k)
    base_evidence = IntegratedGradientsRegionsProvider(
        grid_h=grid_h,
        grid_w=grid_w,
        steps=args.ig_steps,
        baseline=args.ig_baseline,
    )
    interaction = get_interaction(
        flag=args.interaction,
        d_model=(args.interaction_dim if args.interaction != "none" else None),
        num_heads=args.interaction_heads,
        dim_feedforward=args.interaction_ffn,
        dropout=args.interaction_dropout,
    )
    objective = ContrastiveObjective(
        lambda_suff=1.0,
        lambda_margin=game_cfg.lambda_margin,
        lambda_sparse=0.05,
        lambda_overlap=game_cfg.lambda_overlap,
        lambda_mass=args.lambda_mass,
        attn_weight_blend=args.interaction_weight_blend,
    )
    allocator = OptimizationAllocator(
        objective,
        num_steps=args.num_alloc_steps,
        lr=0.2,
        use_shared=game_cfg.use_shared,
        lambda_disjoint=game_cfg.lambda_disjoint,
        lambda_partition=game_cfg.lambda_partition,
        attn_mix=args.interaction_attn_mix,
    )

    explainer = CDEAExplainer(
        model=model,
        unit_space=unit_space,
        selector=selector,
        base_evidence=base_evidence,
        allocator=allocator,
        objective=objective,
        interaction=interaction,
        normalize_evidence=True,
        device=device,
    )

    explanation = explainer.explain(x)
    print("\n--- Contrastive explanation (IG evidence) ---")
    print("Hypotheses (top-%d class ids) shape:" % top_k, explanation.hypotheses.ids.shape)
    print("Unique masks shape:", explanation.masks["unique"].shape)
    if "shared" in explanation.masks:
        print("Shared mask shape:", explanation.masks["shared"].shape)
    probs_top = explanation.extras.get("probs")
    if isinstance(probs_top, torch.Tensor):
        pred_ids = probs_top.argmax(dim=-1).detach().cpu()
    else:
        pred_ids = explanation.hypotheses.ids[:, 0].detach().cpu()
    y0 = int(y_true[0].detach().cpu().item())
    p0 = int(pred_ids[0].item())
    y0_lbl = class_names[y0] if 0 <= y0 < len(class_names) else str(y0)
    p0_lbl = class_names[p0] if 0 <= p0 < len(class_names) else str(p0)
    print("Sample 0: true=%s (%d)  predicted=%s (%d)" % (y0_lbl, y0, p0_lbl, p0))
    print("Metrics: suff=%.4f margin=%.4f overlap=%.4f sparse=%.4f" % (
        explanation.metrics["suff"].item(),
        explanation.metrics["margin"].item(),
        explanation.metrics["overlap"].item(),
        explanation.metrics["sparse"].item(),
    ))
    split_shared = explanation.metrics.get("split_shared_only_probs_topm")
    split_plus = explanation.metrics.get("split_shared_plus_unique_probs_topm")
    if isinstance(split_shared, torch.Tensor) and isinstance(split_plus, torch.Tensor):
        ids0 = explanation.hypotheses.ids[0].detach().cpu()
        valid0 = explanation.hypotheses.mask[0].detach().cpu()
        split_shared0 = split_shared[0].detach().cpu()
        split_plus0 = split_plus[0].detach().cpu()
        print("Probability split report (sample 0, top-m):")
        for k in range(int(valid0.sum().item())):
            cls_id = int(ids0[k].item())
            lbl = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            p_sh = float(split_shared0[k].item())
            p_pl = float(split_plus0[k].item())
            print("  %s: shared-only=%.4f shared+unique=%.4f delta=%+.4f" % (lbl, p_sh, p_pl, p_pl - p_sh))

    out_path = REPO / "examples" / "out" / f"contrastive_explanation_ig_{args.dataset}.png"
    visualize_contrastive(
        sample_idx=0,
        x=x.detach().cpu(),
        explanation=explanation,
        unit_space=unit_space,
        class_names=class_names,
        y_true=y_true.detach().cpu(),
        evidence=explanation.extras.get("evidence"),
        probs=explanation.extras.get("probs"),
        max_viz_classes=args.max_viz_classes,
        out_path=out_path,
    )
    save_contrastive_reports(
        explanation=explanation,
        class_names=class_names,
        dataset=args.dataset,
        out_dir=REPO / "examples" / "out",
        run_name=f"contrastive_ig_{args.dataset}",
        config={
            "evidence": "integrated_gradients",
            "dataset": args.dataset,
            "model": args.model,
            "pretrained": bool(args.pretrained),
            "batch_size": int(args.batch_size),
            "max_viz_classes": int(args.max_viz_classes),
            "num_alloc_steps": int(args.num_alloc_steps),
            "interaction": args.interaction,
            "interaction_dim": int(args.interaction_dim),
            "interaction_heads": int(args.interaction_heads),
            "interaction_ffn": int(args.interaction_ffn),
            "interaction_dropout": float(args.interaction_dropout),
            "interaction_attn_mix": float(args.interaction_attn_mix),
            "interaction_weight_blend": float(args.interaction_weight_blend),
            "game_mode": game_cfg.mode,
            "use_shared": bool(game_cfg.use_shared),
            "lambda_margin": float(game_cfg.lambda_margin),
            "lambda_overlap": float(game_cfg.lambda_overlap),
            "lambda_disjoint": float(game_cfg.lambda_disjoint),
            "lambda_partition": float(game_cfg.lambda_partition),
            "lambda_mass": float(args.lambda_mass),
            "ig_steps": int(args.ig_steps),
            "ig_baseline": args.ig_baseline,
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
