# Smart Traffic Light Controller

A Computational Intelligence (CI) course project: an intelligent traffic light controller for a single intersection, built using **Fuzzy Logic** and tuned with two metaheuristic optimization algorithms — **Particle Swarm Optimization (PSO)** and **Ant Colony Optimization (ACO)**.

## Overview

One intersection, two roads, random car arrivals. A **Mamdani fuzzy controller** decides green-light duration for each road based on current queue lengths, aiming to minimize:

- **W** — average car waiting time
- **Q** — average queue length
- **S** — number of unnecessary stops

Combined into one cost function:

```
C = α·W + β·Q + γ·S
```

**PSO** and **ACO** independently tune the fuzzy controller's parameters to minimize C, and are then compared against each other and against a hand-tuned baseline.

## Structure

```
smart-traffic-light-controller/
├── data/simulation_logs/         generated run data
├── src/
│   ├── simulation/                Phase 1 — traffic simulation      ✅
│   ├── fuzzy/                     Phase 2 — fuzzy controller        ✅
│   ├── optimization/              Phase 3 — pso.py, aco.py
│   └── cost_function.py           Phase 2 — C = αW + βQ + γS         ✅
├── experiments/                   run/test scripts
├── results/plots/, results/tables/  charts and comparison tables
├── notebooks/                     optional exploration
└── report/                        final written report
```

Each `src/` submodule has its own README with full details — start there for implementation specifics. `src/fuzzy/README.md` is written specifically for whoever picks up Phase 3, describing the exact interface to call.

## Pipeline — 4 Phases

| Phase | What | Status |
|---|---|---|
| 1 | Traffic simulation (`TrafficEnv`) — see `src/simulation/README.md` | ✅ Done |
| 2 | Fuzzy controller (Mamdani, centroid defuzzification) + cost function — see `src/fuzzy/README.md` | ✅ Done |
| 3 | PSO + ACO tuning of the fuzzy controller | ⬜ Not started |
| 4 | Comparison, analysis, report, presentation video | ⬜ Not started |

## Team Split

- **Member 1:** Phase 1 (done) → Phase 2 (done)
- **Member 2:** Phase 3 — ready to start now, interface is locked (see `src/fuzzy/README.md`)
- **Both:** Phase 4

## Setup & Running

```bash
pip install -r requirements.txt
python experiments/test_traffic_env.py       # Phase 1 smoke test
python experiments/test_fuzzy_controller.py  # Phase 2 smoke test
```

## Authors

Course project — Computational Intelligence, Semester project.