# Cascaded Tanks Data

This folder contains the benchmark file used by the Cascaded Tanks experiments:

```text
dataBenchmark.mat
```

Expected variables:

```text
uEst, yEst, uVal, yVal, Ts
```

Run `scripts/run_table1_cascaded_tanks.m` from the repository root to reproduce the table-style numerical results.


The Cascaded Tanks data are from https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks. The included `dataBenchmark.mat` contains the estimation and validation splits used in the manuscript. No additional normalization was applied except the smoothing operations documented in `docs/experiment_protocol.md`.
