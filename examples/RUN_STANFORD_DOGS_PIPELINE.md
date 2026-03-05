# Stanford Dogs Pipeline Runbook

Use these commands to retrain the model and run both explanation modes.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
```

## 1) Retrain (and run Grad-CAM explanation at end)

```bash
PYTHONPATH=. python /Users/ssuresh/gambit/examples/contrastive_explanation.py \
  --dataset stanford_dogs \
  --model resnet18 \
  --train \
  --epochs 10 \
  --batch_size 64 \
  --num_alloc_steps 25 \
  --max_viz_classes 3 \
  --checkpoint /Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt
```

## 2) Run Grad-CAM explanation from trained checkpoint

```bash
PYTHONPATH=. python /Users/ssuresh/gambit/examples/contrastive_explanation.py \
  --dataset stanford_dogs \
  --model resnet18 \
  --checkpoint /Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt \
  --batch_size 8 \
  --num_alloc_steps 25 \
  --max_viz_classes 3
```

## 3) Run Integrated Gradients explanation from same checkpoint

```bash
PYTHONPATH=. python /Users/ssuresh/gambit/examples/contrastive_explanation_ig.py \
  --dataset stanford_dogs \
  --model resnet18 \
  --checkpoint /Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt \
  --batch_size 8 \
  --num_alloc_steps 25 \
  --ig_steps 8 \
  --max_viz_classes 3
```

## Output locations

- Checkpoint: `/Users/ssuresh/gambit/examples/out/checkpoints/stanford_dogs_resnet18.pt`
- Grad-CAM figure: `/Users/ssuresh/gambit/examples/out/contrastive_explanation_stanford_dogs.png`
- IG figure: `/Users/ssuresh/gambit/examples/out/contrastive_explanation_ig_stanford_dogs.png`
- Metrics and tables: `/Users/ssuresh/gambit/examples/out/contrastive*_stanford_dogs_*.json|.csv`
