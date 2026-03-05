# How To Add a New CDEA Game Instantiation

This guide is a step-by-step playbook to implement a **new instantiation** in this repo.

## 1) Decide the game formulation first

Before coding, define these clearly:

1. **Players / masks**: what masks are optimized (example: `unique/shared`, `robust/shortcut`, or your own).
2. **Objective terms**: what should increase/decrease under interventions.
3. **Data context**: single input batch `x` or multi-env batch `env`.
4. **Target signal**: predicted class, provided class label, or custom score.
5. **Metrics to report**: scalar metrics and optional per-sample tensors.

If this is not fixed, implementation becomes unstable quickly.

## 2) Pick the module locations

Create a new folder:

- `/Users/ssuresh/gambit/instantiations/<your_game>/`

Minimum files:

1. `objective.py`
2. `allocator.py`
3. Optional `env.py` (if you need environment variants like ID/OOD)
4. Optional `README.md` in that folder

Also add:

1. Example runner script in `/Users/ssuresh/gambit/scripts/` or `/Users/ssuresh/gambit/examples/`
2. Tests in `/Users/ssuresh/gambit/tests/`

## 3) Implement objective first

Follow `AllocationObjective` signature from:

- `/Users/ssuresh/gambit/core/objective.py`

Your class must expose:

```python
def compute(
    self,
    x,
    model,
    unit_space,
    hypotheses,
    masks,
    evidence,
    tokens=None,
    attn=None,
    env=None,
    **kwargs,
):
    ...
    return {"loss": loss, ...metrics...}
```

Rules:

1. Return a scalar tensor as `loss`.
2. Keep metric names stable and explicit (for CSV/JSON reporting).
3. Use `unit_space.keep(...)` / `unit_space.remove(...)` for faithfulness tests.
4. If you use `env`, make behavior explicit when `env is None`.

## 4) Implement allocator second

Follow `Allocator` signature from:

- `/Users/ssuresh/gambit/core/allocator.py`

Typical pattern:

1. Initialize mask logits from `evidence` (or zeros).
2. Optimize with Adam for `num_steps`.
3. Map logits to `[0,1]` via sigmoid.
4. Call your objective each step.
5. Add regularization/constraints (disjointness, partition, sparsity).
6. Return final mask dict with stable key names.

Return format examples:

1. `{"unique": m_unique}`  
2. `{"unique": m_unique, "shared": m_shared}`  
3. `{"robust": m_rob, "shortcut": m_sho}`

## 5) Decide whether you need env batches

If your game uses shift/multi-context:

1. Create `EnvBatch(xs=[...], env_ids=[...])` from `/Users/ssuresh/gambit/core/types.py`
2. Add helper generators in your new `env.py`.
3. Validate that model behavior differs across envs where expected.

If not needed, keep `env=None`.

## 6) Reuse kernel components instead of rewriting orchestration

Use existing kernel:

- `/Users/ssuresh/gambit/core/runner.py` (`CDEAExplainer`)

Wire your instantiation with:

1. `unit_space` (usually `VisionGridUnitSpace`)
2. `selector` (`TopMSelector`)
3. `base_evidence` (Grad-CAM or IG provider)
4. `allocator` (your new one)
5. `objective` (your new one)
6. optional `interaction` from `/Users/ssuresh/gambit/core/interaction.py`

## 7) Add an example runner script

Create:

- `/Users/ssuresh/gambit/scripts/eval_<your_game>.py`

Must include:

1. Dataset loading
2. Model setup (and checkpoint loading if needed)
3. Explainer wiring
4. `explain(...)` call
5. Saved artifacts:
   - `scripts/out/<your_game>_metrics.json`
   - `scripts/out/<your_game>_metrics.csv`
   - optional per-batch CSV and PNG

Use reporting helpers from:

- `/Users/ssuresh/gambit/core/reporting.py`

## 8) Add tests immediately

Create:

- `/Users/ssuresh/gambit/tests/test_<your_game>.py`

Minimum test cases:

1. Allocator returns expected mask keys/shapes.
2. Objective returns finite `loss` and metric tensors.
3. No crash when running one `CDEAExplainer.explain(...)` pass.
4. At least one directional sanity check (example: overlap decreases, gap positive, etc.).

Recommended to run in your env:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate marl
cd /Users/ssuresh/gambit
PYTHONPATH=. python /Users/ssuresh/gambit/tests/test_<your_game>.py
```

## 9) Practical design considerations (important)

1. **Intervention baseline choice** (`mean` vs `blur`) changes metric behavior.
2. **Mask collapse risk**: all-zero/all-one masks can happen without balanced penalties.
3. **Metric leakage**: always compute metrics via interventions, not raw evidence.
4. **Mass control**: use mass/partition penalties if masks drift.
5. **Top-K dependence**: objective behavior changes with hypothesis count `K`.
6. **Compute cost**: per-step objective with model forward passes is expensive; profile early.
7. **Interaction optionality**: keep your instantiation valid when `interaction=None`.
8. **Reproducibility**: lock seeds and log full config in saved JSON.

## 10) Definition of done checklist

A new instantiation is "done" when all are true:

1. New `objective.py` and `allocator.py` implemented in `instantiations/<your_game>/`.
2. Example script runs end-to-end and saves metrics/artifacts.
3. Tests pass in `marl`.
4. Metrics are interpretable (not all near-zero or degenerate).
5. README/docs contain run command and metric definitions.

## 11) Starter skeleton (copy and adapt)

```python
# /Users/ssuresh/gambit/instantiations/<your_game>/objective.py
class YourGameObjective:
    def __init__(self, ...):
        ...

    def compute(self, x, model, unit_space, hypotheses, masks, evidence, tokens=None, attn=None, env=None, **kwargs):
        # 1) interventions
        # 2) score terms
        # 3) penalties
        loss = ...
        return {
            "loss": loss,
            "metric_a": ...,
            "metric_b": ...,
        }
```

```python
# /Users/ssuresh/gambit/instantiations/<your_game>/allocator.py
class YourGameAllocator:
    def __init__(self, objective, num_steps=40, lr=0.3, ...):
        ...

    def allocate(self, x, model, unit_space, hypotheses, evidence, tokens=None, attn=None, env=None, **kwargs):
        # 1) init mask logits
        # 2) optimize
        # 3) return dict of masks
        return {"mask_key": ...}
```

## 12) Good next move after implementation

After first successful run, do one ablation immediately:

1. no interaction vs attention vs transformer (if applicable)
2. base evidence provider A vs B (Grad-CAM vs IG)
3. 2-3 regularization settings to test stability

This quickly tells you whether the new game is structurally valid or just numerically fitting noise.

