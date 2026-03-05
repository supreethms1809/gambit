# CDEA In Jupyter Notebook

This guide shows how to run contrastive CDEA directly in a notebook and switch base evidence between Grad-CAM and Integrated Gradients (IG).

## 1) Start Jupyter in the right environment

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
jupyter lab
```

## 2) Notebook setup cell

```python
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

REPO = Path("/Users/ssuresh/gambit")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.runner import CDEAExplainer
from core.hypotheses import TopMSelector
from core.device import get_device
from core.interaction import get_interaction
from modality.grid_regions import VisionGridUnitSpace
from instantiations.contrastive.objective import ContrastiveObjective
from instantiations.contrastive.allocator import OptimizationAllocator
from base_evidence.gradcam_regions import GradCAMRegionsProvider
from base_evidence.integrated_gradients_regions import IntegratedGradientsRegionsProvider
from examples.contrastive_explanation import (
    TV_INPUT_SIZE,
    _load_batch_with_labels,
    get_torchvision_model,
    visualize_contrastive,
)
```

## 3) Config cell

```python
dataset = "cifar10"            # mnist | cifar10 | pets | stanford_dogs
batch_size = 4
model_name = "resnet18"
pretrained = False
num_alloc_steps = 25
evidence_mode = "gradcam"      # gradcam | ig

interaction_mode = "none"      # none | attention | transformer
interaction_dim = 64
interaction_heads = 4
interaction_ffn = 256
interaction_dropout = 0.0

ig_steps = 24
ig_baseline = "zero"           # zero | mean

use_shared = True
lambda_partition = 0.1
lambda_mass = 2.0
grid_h, grid_w = 7, 7
```

## 4) Build model + data cell

```python
device = get_device()
x, y_true, class_names, num_classes = _load_batch_with_labels(dataset, batch_size, device)
if x.shape[-1] != TV_INPUT_SIZE or x.shape[-2] != TV_INPUT_SIZE:
    x = F.interpolate(x, size=(TV_INPUT_SIZE, TV_INPUT_SIZE), mode="bilinear", align_corners=False)

model = get_torchvision_model(model_name, num_classes, pretrained=pretrained).to(device).eval()
top_k = min(5, num_classes)
```

## 5) Build explainer cell

```python
embed_dim = interaction_dim if interaction_mode != "none" else None
unit_space = VisionGridUnitSpace(grid_h, grid_w, baseline="blur", embed_dim=embed_dim)
selector = TopMSelector(m=top_k)

if evidence_mode == "gradcam":
    base_evidence = GradCAMRegionsProvider(grid_h, grid_w)
elif evidence_mode == "ig":
    base_evidence = IntegratedGradientsRegionsProvider(
        grid_h=grid_h,
        grid_w=grid_w,
        steps=ig_steps,
        baseline=ig_baseline,
    )
else:
    raise ValueError("evidence_mode must be 'gradcam' or 'ig'")

interaction = get_interaction(
    flag=interaction_mode,
    d_model=(interaction_dim if interaction_mode != "none" else None),
    num_heads=interaction_heads,
    dim_feedforward=interaction_ffn,
    dropout=interaction_dropout,
)

objective = ContrastiveObjective(
    lambda_suff=1.0,
    lambda_margin=1.0,
    lambda_sparse=0.05,
    lambda_overlap=0.2,
    lambda_mass=lambda_mass,
)
allocator = OptimizationAllocator(
    objective=objective,
    num_steps=num_alloc_steps,
    lr=0.2,
    use_shared=use_shared,
    lambda_disjoint=0.1,
    lambda_partition=lambda_partition,
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
```

## 6) Run + visualize cell

```python
explanation = explainer.explain(x)
print({
    "true_class_sample0": class_names[int(y_true[0].item())],
    "pred_class_sample0": class_names[int(explanation.extras["probs"][0].argmax().item())],
    "suff": float(explanation.metrics["suff"].item()),
    "margin": float(explanation.metrics["margin"].item()),
    "overlap": float(explanation.metrics["overlap"].item()),
    "sparse": float(explanation.metrics["sparse"].item()),
})

out = REPO / "examples" / "out" / f"notebook_{evidence_mode}_{dataset}.png"
visualize_contrastive(
    sample_idx=0,
    x=x.detach().cpu(),
    explanation=explanation,
    unit_space=unit_space,
    class_names=class_names,
    y_true=y_true.detach().cpu(),
    evidence=explanation.extras.get("evidence"),
    probs=explanation.extras.get("probs"),
    max_viz_classes=3,
    out_path=out,
)
out
```

## 7) Presentation cell (top-m probability split table)

```python
import pandas as pd

ids0 = explanation.hypotheses.ids[0].detach().cpu()
valid0 = explanation.hypotheses.mask[0].detach().cpu()
split_shared = explanation.metrics.get("split_shared_only_probs_topm")
split_plus = explanation.metrics.get("split_shared_plus_unique_probs_topm")

rows = []
if isinstance(split_shared, torch.Tensor) and isinstance(split_plus, torch.Tensor):
    sh0 = split_shared[0].detach().cpu()
    pl0 = split_plus[0].detach().cpu()
    for k in range(int(valid0.sum().item())):
        cid = int(ids0[k].item())
        lbl = class_names[cid] if cid < len(class_names) else str(cid)
        rows.append({
            "rank": k,
            "class_id": cid,
            "label": lbl,
            "p_shared_only": float(sh0[k].item()),
            "p_shared_plus_unique": float(pl0[k].item()),
            "delta": float(pl0[k].item() - sh0[k].item()),
        })

pd.DataFrame(rows)
```

## 8) Pairwise "why k rather than l" table

```python
pair_shared = explanation.metrics.get("pairwise_margin_shared_only_logits_topm")
pair_plus = explanation.metrics.get("pairwise_margin_shared_plus_unique_logits_topm")
pair_delta = explanation.metrics.get("pairwise_margin_delta_logits_topm")

pair_rows = []
if isinstance(pair_shared, torch.Tensor) and isinstance(pair_plus, torch.Tensor) and isinstance(pair_delta, torch.Tensor):
    ps = pair_shared[0].detach().cpu()
    pp = pair_plus[0].detach().cpu()
    pd = pair_delta[0].detach().cpu()
    for k in range(int(valid0.sum().item())):
        for l in range(int(valid0.sum().item())):
            if k == l:
                continue
            cid_k = int(ids0[k].item())
            cid_l = int(ids0[l].item())
            pair_rows.append({
                "k_rank": k,
                "l_rank": l,
                "k_label": class_names[cid_k] if cid_k < len(class_names) else str(cid_k),
                "l_label": class_names[cid_l] if cid_l < len(class_names) else str(cid_l),
                "shared_only_margin_logit": float(ps[k, l].item()),
                "shared_plus_unique_margin_logit": float(pp[k, l].item()),
                "delta_margin_logit": float(pd[k, l].item()),
            })

pd.DataFrame(pair_rows).sort_values("delta_margin_logit", ascending=False)
```

## Notes

- For interpretable masks, use a trained checkpoint rather than a random model.
- IG is slower than Grad-CAM (multiple backward passes controlled by `ig_steps`).
- Keep `interaction_mode="none"` to match default behavior; enable attention/transformer when running interaction ablations.
