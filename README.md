# Functional transfer from developmental world-model curricula

This project extends `neurodev_wm` and only reads from it. `neurodev_wm` trains world models under different schedules of experience and measures when each capability emerges. This project freezes those world models and asks a follow-up question. Does the order of a model's earlier experience change how fast a pose-free navigation policy learns from reward on top of it?

No file in `neurodev_wm` is changed. Its checkpoints and `runs/development.json` are inputs. The arena, the world-model classes and the reward-only actor-critic are imported from `robowomo-attractors`, with no copies.

## Why a separate project

`neurodev_wm` already covers emergence order, deprivation timing, seed scaling and the label-free heading measure. A second copy of those analyses would give two sources of truth. Functional transfer measures a consequence that the developmental battery does not: whether a state shaped by the curriculum helps later embodied learning.

The main causal comparison is between `slow_first`, `fast_first` and `shuffled`. These schedules show the models the same marginal mixture of worlds in different temporal orders. `natural` is a reference, but it does not share that mixture. The `deprived_early` and `deprived_late` pair is not a primary contrast. In the current fixed-budget design, the two conditions have unequal recovery time after deprivation.

## Outcomes and unit of inference

The primary outcomes are final goal success after a fixed actor-critic budget, and the area under the goal-success learning curve.

The unit of inference is an architecture and encoder-seed pair. Policy seeds are averaged within each encoder before the paired curriculum tests. Episodes and timesteps are not independent samples.

The source grid now has three seeds per architecture, so the analysis returns the verdict `MORE_SEEDS`. When `neurodev_wm` adds seed 3, the design has eight architecture and seed pairs. The exact sign-flip test then has a floor of 0.0078125, and the analysis becomes confirmatory without code changes.

## Run

Smoke test:

```bash
PYTHONPATH=src python3 -m pytest -q

PYTHONPATH=src python3 src/train_transfer.py \
  --curricula natural --kinds gru --encoder-seeds 0 --policy-seeds 0 \
  --updates 20 --batch 16 --outdir runs/smoke

PYTHONPATH=src python3 src/analyze_transfer.py \
  --indir runs/smoke --required-policy-seeds 0 \
  --output runs/smoke_analysis.json --allow-incomplete
```

Confirmatory run, with three policy seeds for the order-matched curricula:

```bash
for curriculum in slow_first fast_first shuffled; do
  OMP_NUM_THREADS=4 PYTHONPATH=src python3 src/train_transfer.py \
    --curricula "$curriculum" --kinds gru,rnn \
    --encoder-seeds 0,1,2,3 --policy-seeds 0,1,2 \
    --updates 1000 --batch 128 --outdir runs/confirmatory &
done
wait

PYTHONPATH=src python3 src/analyze_transfer.py \
  --indir runs/confirmatory --required-policy-seeds 0,1,2 \
  --output runs/transfer_analysis.json
```

If seed 3 is missing from `neurodev_wm/runs/dev`, the training command stops with a report of the missing cells. It does not shrink the design without warning.

The code expects `neurodev_wm` and `robowomo-attractors` as sibling folders. See `src/links.py` and `configs/confirmatory.json` for the paths.

## Layout

    src/links.py              read-only imports and source-path checks
    src/train_transfer.py     policy learning on frozen developmental states
    src/analyze_transfer.py   encoder-level aggregation and paired tests
    configs/confirmatory.json frozen design matrix and claim gates
    tests/                    tag, aggregation and completeness invariants
    runs/                     outputs of this project only (smoke run so far)

Policy checkpoints (`*.pt`) are not in the repository.

## Limits on claims

- The result is functional transfer from simulated curricula. It says nothing about development in a biological organism.
- The policy learns from reward on a frozen representation. The setup is not joint end-to-end reinforcement learning.
- Correlations between emergence times and policy results are descriptive, unless a separate clustered inferential design is preregistered.
- The early and late deprivation endpoint is not interpreted, because recovery time is confounded in the source experiment.
- This result may suit a follow-on paper or an appendix analysis better than the parent manuscript.
