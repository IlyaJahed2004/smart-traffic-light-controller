# Smart Traffic Light Controller

A Computational Intelligence (CI) course project: designing and implementing an intelligent traffic light controller for a single urban intersection using **Fuzzy Logic** and two metaheuristic optimization algorithms — **Particle Swarm Optimization (PSO)** and **Ant Colony Optimization (ACO)**.

## Overview

The project simulates a single intersection with two roads, each controlled by its own traffic light. Cars arrive randomly on each road. A **Mamdani-type fuzzy controller** decides the green-light duration for each road based on current queue lengths, aiming to minimize:

- Average car waiting time (W)
- Average queue length (Q)
- Number of unnecessary stops (S)

These are combined into a single cost function:

```
C = α·W + β·Q + γ·S
```

Since hand-tuned fuzzy membership functions and rules are rarely optimal, this project uses **PSO** and **ACO** to automatically tune the fuzzy controller's parameters, minimizing the cost function above. The two optimization algorithms are then compared on convergence speed, final cost, and stability.

## Project Structure

```
smart-traffic-light-controller/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── simulation_logs/          # generated data from simulation runs
├── src/
│   ├── simulation/
│   │   └── traffic_env.py        # discrete-time traffic simulation
│   ├── fuzzy/
│   │   └── fuzzy_controller.py   # Mamdani fuzzy controller
│   ├── optimization/
│   │   ├── pso.py                # Particle Swarm Optimization
│   │   └── aco.py                # Ant Colony Optimization
│   ├── cost_function.py          # C = αW + βQ + γS
│   └── utils.py
├── experiments/
│   ├── run_baseline_fuzzy.py     # fuzzy controller with hand-set params
│   ├── run_pso_optimization.py
│   ├── run_aco_optimization.py
│   └── compare_algorithms.py
├── results/
│   ├── plots/                    # convergence curves, comparison charts
│   └── tables/                   # comparison tables
├── notebooks/
│   └── exploration.ipynb         # optional interactive exploration
└── report/
    └── report.pdf                # final written report
```

## Pipeline

1. **Traffic simulation** — discrete-time model of car arrivals, queues, and departures at a two-road intersection.
2. **Fuzzy controller** — Mamdani inference with 2 inputs (queue length road 1, queue length road 2), 1 output (green time road 1), each with `Low/Medium/High` (inputs) and `Short/Medium/Long` (output) fuzzy sets; centroid defuzzification.
3. **Optimization** — PSO and ACO independently search for fuzzy parameters (membership function shapes and/or rule weights) that minimize the cost function.
4. **Comparison** — evaluate PSO vs. ACO vs. baseline fuzzy controller on final cost, convergence speed, and run-to-run stability.

## Status

🚧 Work in progress — being built incrementally, module by module.

| Module | Status |
|---|---|
| Repo structure | ✅ Done |
| Traffic simulation | ⬜ Not started |
| Fuzzy controller | ⬜ Not started |
| PSO optimization | ⬜ Not started |
| ACO optimization | ⬜ Not started |
| Comparison & analysis | ⬜ Not started |
| Report | ⬜ Not started |

## Requirements

See `requirements.txt` (to be populated as dependencies are added — likely `numpy`, `matplotlib`, `scikit-fuzzy` or a custom fuzzy implementation).

## Authors

Course project — Computational Intelligence, Semester project.