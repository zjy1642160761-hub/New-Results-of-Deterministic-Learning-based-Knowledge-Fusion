# Experiment Protocol

## Cascaded Tanks

The script `scripts/run_table1_cascaded_tanks.m` runs five method labels:

```text
DLM-Mean
DLM-Off
DLM-LS
DLM-WLSR1
DLM-WLSR2
```

Each label maps to one original legacy script and preserves its hyperparameters. The shared workflow is:

1. Load `data/cascaded_tanks/dataBenchmark.mat`.
2. Smooth estimation signals using the same spans as the original code.
3. Split the estimation trajectory into method-specific segments.
4. Learn deterministic-learning weights with `DL_4D`.
5. Fuse segment-level weights with the method-specific fusion function.
6. Run free-run prediction on `yEst/uEst` and `yVal/uVal`.
7. Save RMSE metrics to `results/generated/table1_cascaded_tanks_results.csv`.

Important: the smoothing windows are method-specific in the original scripts and must not be replaced by a single global default. The refactored runner preserves those per-script values.

## Rossler Main Figures

`scripts/run_rossler_main_figures.m` runs the original `scripts/legacy/NNpaperfigure.m` script with organized paths and exports all open figures.

The main Rossler protocol:

1. Generate six trajectories with different `p3` values.
2. Learn the third dynamics component with deterministic learning.
3. Fuse learned weights by weighted, pseudo-inverse, least-squares, and mean methods.
4. Evaluate trajectory-level approximation error.
5. Evaluate grid-level surface generalization error.

## Surface Error Filter

The original Rossler code computes:

```matlab
mean(Z(Z < 0.3))
```

This means the reported surface-level error is averaged over a valid region where the pointwise relative error is less than `0.3`. The rule is preserved for traceability. If reviewers ask for an unfiltered mean, compute `mean(Z(:))` and report both metrics.
