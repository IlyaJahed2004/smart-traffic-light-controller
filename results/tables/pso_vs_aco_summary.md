# PSO vs ACO — comparison summary

## 1. Final cost-function value

Baseline (default hand-picked params): **73.6823**

| Method | Best avg cost | Improvement over baseline | Runtime (s) |
|---|---|---|---|
| PSO | 68.1014 | +5.5809 (+7.6%) | 611.54 |
| ACO | 72.5857 | +1.0966 (+1.5%) | 744.89 |

## 2. Convergence speed

See `results/plots/pso_vs_aco_convergence.png` for the full curve. Best-cost-so-far sampled across the run:

- **PSO**: [73.682, 70.364, 69.119, 68.684, 68.266, 68.114, 68.102, 68.102, 68.102]
- **ACO**: [85.408, 81.037, 77.558, 75.195, 75.195, 73.492, 73.492, 72.586, 72.586]

## 3. Stability of the solution across different runs

Each optimizer re-run independently with seeds [1, 2, 3, 4, 5] on the identical multi-scenario fitness landscape. Every individual run was validated against the baseline (see module docstring) before being included here.

| Method | Cost mean | Cost std | Cost min | Cost max | Mean per-parameter std |
|---|---|---|---|---|---|
| PSO | 68.0614 | 0.6028 | 66.9763 | 68.8265 | 1.1566 |
| ACO | 73.7787 | 1.0795 | 72.3515 | 75.4254 | 3.9225 |

Lower `cost std` / `mean per-parameter std` = more stable (lands on a similar answer regardless of random seed).


## 4. Effect of optimization on the fuzzy controller's performance

Baseline (untuned) vs. each tuned controller, evaluated on every individual traffic scenario/seed, not just the aggregate:

| Scenario | Baseline | PSO | ACO |
|---|---|---|---|
| moderate symmetric (seed=1) | 45.72 | 48.26 | 45.77 |
| moderate symmetric (seed=2) | 52.68 | 49.66 | 48.31 |
| moderate symmetric (seed=3) | 48.94 | 47.95 | 44.70 |
| moderate asymmetric (seed=1) | 43.69 | 43.56 | 43.27 |
| moderate asymmetric (seed=2) | 44.97 | 46.03 | 44.20 |
| moderate asymmetric (seed=3) | 41.47 | 42.65 | 39.72 |
| heavy asymmetric (seed=1) | 54.17 | 54.84 | 59.74 |
| heavy asymmetric (seed=2) | 57.07 | 62.33 | 74.01 |
| heavy asymmetric (seed=3) | 53.30 | 57.24 | 67.93 |
| heavy symmetric (seed=1) | 132.43 | 111.39 | 119.65 |
| heavy symmetric (seed=2) | 155.05 | 126.53 | 142.23 |
| heavy symmetric (seed=3) | 154.69 | 126.79 | 141.50 |