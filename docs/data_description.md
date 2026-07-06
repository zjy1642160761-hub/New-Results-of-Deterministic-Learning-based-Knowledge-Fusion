# Data Description

## Cascaded Tanks

Data file:

```text
data/cascaded_tanks/dataBenchmark.mat
```

Variables:

| Variable | Size | Description |
|---|---:|---|
| `uEst` | 1024 x 1 | Input signal for the estimation set |
| `yEst` | 1024 x 1 | Output signal for the estimation set |
| `uVal` | 1024 x 1 | Input signal for the validation set |
| `yVal` | 1024 x 1 | Output signal for the validation set |
| `Ts` | 1 x 1 | Sampling time |

The scripts use `uEst` and `yEst` for deterministic learning and report free-run prediction performance on both the estimation and validation sets.

## Rossler System

Rossler trajectories are generated directly in MATLAB. The main scripts use parameter triples:

```text
[0.2, 0.4, 2.05]
[0.2, 0.4, 1.95]
[0.2, 0.4, 1.85]
[0.2, 0.4, 1.75]
[0.2, 0.4, 1.65]
[0.2, 0.4, 1.55]
```

No external Rossler data file is required.
