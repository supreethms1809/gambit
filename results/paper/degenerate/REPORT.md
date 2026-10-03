# Degenerate-optimum checks

Val split only. Seeded subset. The thresholds are the constraints named in `evaluation/degenerate.py`.

- D1 closed when the shared-mask fraction above 0.5 is at most 0.15 and the mass fraction is at most 0.15.
- D2 closed when the soft kept logit exceeds the same-area hard mask by at most 0.5.
- D3 closed when the share of a risen margin coming from foil suppression is at most 0.5.
- D4 closed when unique-mask evidence capture is at least 1.2 times chance and above a translated copy.
- D5 closed when unique-mask mass over the mass target is at most 1.1.
- D6 closed when the robust-mask fraction above 0.5 is at most 0.25, and the shortcut mask is not a copy of the robust complement (mean absolute deviation at least 0.25, or shortcut mass fraction at most 0.5).
- D7 closed by recording the ID-OOD gap as a model property on the full image, in `scripts/eval_robust_shortcut.py` and `scripts/run_experiments.py`.

## cifar10

Checkpoint `results/paper_rerun/checkpoints/cifar10_resnet18_pt_lp_ep15_lr0.001_seed0.pt`. n=48. Val accuracy on this subset: 0.729. Allocator steps: 50.

| Route | Status | Measurement |
|---|---|---|
| D1 shared blanket | open | fraction above 0.5 = 0.1122, mass fraction = 0.1616, evidence capture / area = 1.0873 |
| D2 soft vs hard | open | soft minus hard logit = 1.2245 |
| D3 foil suppression | closed | suppression share = 0.4249, z_k keep/full = 3.772/2.400, z_l keep/full = -1.754/-0.568 |
| D4 arbitrary cells | closed | capture ratio = 1.4273, translated null = 0.9415 |
| D5 budget | open | mass ratio = 5.1980 |
| D6 shift masks | closed | robust fraction above 0.5 = 0.1348, complement deviation = 0.6864, shortcut mass fraction = 0.1426 |
| D7 model gap | closed | moved out of the method table |

## ham10000

Checkpoint `examples/out/checkpoints/ham10000_resnet18.pt`. n=48. Val accuracy on this subset: 0.854. Allocator steps: 50.

| Route | Status | Measurement |
|---|---|---|
| D1 shared blanket | closed | fraction above 0.5 = 0.0196, mass fraction = 0.0444, evidence capture / area = 1.8680 |
| D2 soft vs hard | closed | soft minus hard logit = 0.3984 |
| D3 foil suppression | open | suppression share = 0.7647, z_k keep/full = 5.877/5.990, z_l keep/full = 1.454/2.610 |
| D4 arbitrary cells | closed | capture ratio = 1.7714, translated null = 0.8599 |
| D5 budget | open | mass ratio = 4.9308 |
| D6 shift masks | closed | robust fraction above 0.5 = 0.1390, complement deviation = 0.7194, shortcut mass fraction = 0.1055 |
| D7 model gap | closed | moved out of the method table |

Open routes: cifar10 D1, cifar10 D2, cifar10 D5, ham10000 D3, ham10000 D5.

## What stays open

The run above uses the current defaults: `lambda_mass=0.1`, `lambda_shared_sparse=0.25`, mixed presets, 50 Adam steps. A follow-up on 16 val images, same steps, varied only those two weights. Shift was not re-run; D6 was already closed.

| lambda_mass | lambda_shared_sparse | CIFAR D1 mass frac | CIFAR D2 gap | CIFAR D5 ratio | CIFAR kept logit | HAM D3 share | HAM D5 ratio | HAM kept logit |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.25 | 0.185 | 0.968 | 3.144 | 2.15 | 0.797 | 2.190 | 4.67 |
| 1.0 | 0.25 | 0.204 | 0.903 | 1.723 | 1.31 | 0.901 | 1.323 | 3.70 |
| 2.0 | 0.25 | 0.201 | 0.425 | 1.074 | 0.37 | 0.924 | 1.025 | 3.04 |
| 2.0 | 0.50 | 0.071 | 0.357 | 1.126 | -0.57 | 0.916 | 1.059 | 3.05 |

`lambda_mass=2` brings the unique-mask mass ratio to the D5 line and brings the CIFAR soft-versus-hard gap under 0.5. It also drops the CIFAR kept logit from 3.77 at the default to 0.37, and the foil-suppression share rises on both datasets. Raising the shared-mask penalty to 0.5 on top of that closes the CIFAR shared-mass fraction and drives the CIFAR kept logit negative. Those weights are not the new default.

Phase 4 has to select `lambda_mass` and `lambda_shared_sparse` on val under all of these constraints at once: D1 mass fraction at most 0.15, D2 gap at most 0.5, D3 suppression share at most 0.5, D5 mass ratio at most 1.1, and the kept logit not collapsing. No point in this sweep meets that set. D3 on HAM is open at every point tried: the margin moves because the foil logit falls while the kept logit of the predicted class stays flat.
