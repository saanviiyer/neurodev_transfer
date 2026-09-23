# Functional transfer from developmental world-model curricula

This is a read-only extension of `../neurodev_wm`. It asks whether the
representations produced by different developmental experience schedules have
different functional value when a new policy must learn from reward.

The developmental project measures *when* capabilities emerge. This extension
asks a separate question: **after the world model is frozen, does the order of
its earlier experience change how quickly a pose-free navigation policy
learns?**

No file in `neurodev_wm` is modified. Its checkpoints and
`runs/development.json` are inputs. The arena, world-model classes, and
reward-only actor--critic are imported from `robowomo-attractors` rather than
copied.

## Why this is complementary

The active developmental work already owns emergence ordering, deprivation
timing, seed scaling, and the label-free heading instrument. Repeating those
analyses here would create two sources of truth. Functional transfer instead
supplies a consequence that the developmental battery does not measure:
whether curriculum-shaped state improves later embodied learning.

The clean causal comparison is `slow_first` versus `fast_first` versus
`shuffled`. These schedules expose models to the same marginal mixture of
worlds in different temporal orders. `natural` is a useful reference but does
not share that mixture. `deprived_early` versus `deprived_late` is deliberately
not a primary transfer contrast because the current fixed-budget design has
unequal post-deprivation recovery time.

## Primary outcomes and unit of inference

- Final goal success after a fixed actor--critic budget.
- Area under the goal-success learning curve.

The unit is an architecture--encoder-seed pair. Policy seeds are averaged
within each encoder before paired curriculum tests. Episodes and timesteps are
not independent samples.

The current source grid has three seeds per architecture, so a confirmatory
verdict is `MORE_SEEDS`. The analysis becomes confirmatory automatically when
the other project adds seed 3, producing eight architecture--seed pairs and an
exact sign-flip floor of 0.0078125.

## Layout

    src/links.py              read-only imports and source-path validation
    src/train_transfer.py     policy learning on frozen developmental states
    src/analyze_transfer.py   encoder-level aggregation and paired tests
    configs/confirmatory.json frozen matrix and claim gates
    tests/                    tag, aggregation, and completeness invariants
    runs/                     extension-owned outputs only

## Smoke test

```bash
PYTHONPATH=src python3 -m pytest -q

PYTHONPATH=src python3 src/train_transfer.py \
  --curricula natural --kinds gru --encoder-seeds 0 --policy-seeds 0 \
  --updates 20 --batch 16 --outdir runs/smoke

PYTHONPATH=src python3 src/analyze_transfer.py \
  --indir runs/smoke --required-policy-seeds 0 \
  --output runs/smoke_analysis.json --allow-incomplete
```

## Confirmatory run

Run three policy seeds for the order-matched curriculum trio:

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

If seed 3 is not yet present in `neurodev_wm/runs/dev`, the training command
will fail with a missing-cell report instead of silently reducing the design.

## Claim discipline

- Say **functional transfer from simulated curricula**, not development in a
  biological organism.
- Say **reward-only policy learning on a frozen representation**, not joint
  end-to-end reinforcement learning.
- Treat emergence-policy correlations as descriptive unless a separate
  clustered inferential design is preregistered.
- Do not interpret the current early/late deprivation endpoint; recovery time
  is confounded in the source experiment.
- Do not merge this result into the parent ICLR manuscript unless the visual
  replication and core claim remain clear within the page budget. It may be a
  better follow-on paper or appendix analysis.
