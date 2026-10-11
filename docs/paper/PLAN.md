# Plan

This branch is the contrastive method. History before the archive lives on tag `framing-v1`.

## Stages

P1 docs, P2 agent instructions, P3 scope guard, P4 one model builder, P5 eval loaders, P6 shift removed, P7 earlier method removed, P8 earlier-era material removed, P9 one deletion operator, P10 `cdea/`, P11 registry and records, P12 baselines in separate reviewed PRs, P13 scores and completion checks, P14 the G1 pilot.

## Baseline protocol

B1 name the reviewer question. B2 pin the source. B3 record the reproduction target before running it. B4 one conversion, `baselines.adapter.adapt_scores`. B5 failure modes raise or write a failed row. B6 cost is forward and backward counts. B7 a baseline PR is reviewed in a fresh session. B8 do not change a baseline's algorithm except through that review.

## Commits

One logical change per commit. `PYTHONPATH=. python -m pytest tests/` passes before every commit. `git add` names paths. Code and results never share a commit. Do not run `--final` or read the test split before the `eval-plan-frozen` tag. Do not type a result number by hand.
