# Datasets

Medical images use a pre-split `<root>/<split>/<class>/` layout. Grouped splits keep near-duplicate images out of both train and val: by lesion for HAM10000, by patient for brain tumor.

```bash
PYTHONPATH=. python scripts/prepare_ham10000.py
PYTHONPATH=. python scripts/prepare_brain_tumor.py
```

The longer preparation notes are on tag `framing-v1`, in `docs/MEDICAL_DATASETS.md`.
