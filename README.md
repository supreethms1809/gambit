# GAMBIT

Game theoretic Allocation for Model Based Interpretability and Trust.

CDEA allocates an image's evidence between competing hypotheses: the predicted class k and a rival l. It splits the evidence into unique-to-k, unique-to-l and shared, and scores the split by deleting it from the image. The method is in `cdea/` and is specified by `docs/paper/FORMULATION_DECOMPOSITION.md`. The contrast game, its ablation DA2, is `docs/paper/FORMULATION.md`. The evaluation is specified by `docs/paper/EVAL_PLAN.md`, headed by section 13.

## Setup

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda create -n gambit python=3.12 -y
conda activate gambit
pip install -r requirements.txt
```

On NVIDIA, `torch.cuda.is_available()` must be true. If it is not, reinstall `torch==2.9.1` and `torchvision==0.24.1` from the CUDA index that matches `nvidia-smi`. Then install `requirements.txt` again.

Run everything from the repository root with `PYTHONPATH=.`.

## Data and checkpoints

- **Data.** Datasets go under `data/`. The split files are committed in `data/splits/`.
  - HAM10000 and brain tumor MRI are prepared with `scripts/prepare_ham10000.py` and `scripts/prepare_brain_tumor.py`. See `docs/DATASETS.md`.
  - The ImageNet split is written once with `scripts/write_paper_splits.py --only imagenet`.
- **Checkpoints.** Paper checkpoints go in `results/paper/checkpoints/`. Train missing ones with `scripts/launch_paper_training.py`.
- **Test split.** It stays locked until `EVAL_PLAN.md` is frozen and tagged `eval-plan-frozen`. Until then, run `--split val` only.

## Check

```bash
PYTHONPATH=. python -m pytest tests/
```

## Run

One cell is every method on one dataset × backbone × seed. Records go to `results/paper/cells/`.

```bash
# contrastive game, fast knobs, one image
PYTHONPATH=. python scripts/paper_run.py --game contrastive --dataset cifar10 --backbone resnet50 --seed 0 --split val --n 1 --methods core --fast

# decomposition game, three areas
PYTHONPATH=. python scripts/paper_run.py --game decomposition --dataset cifar10 --backbone resnet50 --seed 0 --split val --n 64 --areas 0.025,0.05,0.1
```

Without paper checkpoints, add `--model-source smoke` to check the pipeline. Fast knobs and smoke models never satisfy a gate cell. A record counts only when its `method_code_hash` and `knobs_hash` match the current tree.

The full grid trains each missing checkpoint, then runs that checkpoint's games before moving to the next dataset:

```bash
PYTHONPATH=. python scripts/launch_paper_chain.py --seeds 0 --jobs 4
```

`scripts/smoke_e2e.py` runs every unit and method on a few val images. `docs/paper/SPARK_RUNBOOK.md` gives the order for a full run on the GPU machine.

## Read-outs

- `scripts/decomp_readout.py --records results/paper/cells --split val`: the decomposition game (`EVAL_PLAN.md` section 13).
- `analysis/stats.py`:
  - family C: Wilcoxon, Holm and the bootstrap over datasets;
  - families T and D-T: the within-seed sign-flip test and the two-level bootstrap.

## Layout

| Path | Contents |
|---|---|
| `cdea/` | The method: payoffs, Sinkhorn plan, allocation, decomposition, shift. It imports `torch`, `core` and `base_evidence` only. |
| `core/` | Types, hypotheses, eval mode, device, reporting, and `grid.py` (units and the deletion baseline). |
| `base_evidence/` | The Grad-CAM and integrated-gradients evidence that CDEA starts from. |
| `baselines/` | The comparators. `docs/paper/BASELINES.md` lists each one's pinned source. |
| `evaluation/` | Data, models, methods and scores for one cell. |
| `models/` | ResNet-50 and ViT-B/16. |
| `analysis/` | Statistics and result tables built from records. |
| `scripts/` | Entry points: run, launch, train, read-outs. |
| `third_party/` | Vendored baseline code, pinned. |
| `docs/paper/` | Formulation, evaluation plan, gates, baselines, and `PROGRESS.md` (current state). |

The earlier method is on tag `framing-v1`. Work that needs it uses a worktree checked out at that tag.
