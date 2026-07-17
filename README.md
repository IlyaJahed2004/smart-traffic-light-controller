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

## Pipeline — 4 Phases

| Phase | What | Status |
|---|---|---|
| 1 | Traffic simulation (`TrafficEnv`) — see `src/simulation/README.md` | ✅ Done |
| 2 | Fuzzy controller (Mamdani, centroid defuzzification) + cost function — see `src/fuzzy/README.md` | ✅ Done |
| 3 | PSO + ACO tuning of the fuzzy controller — see `src/optimization/README.md` | ✅ Done |
| 4 | Comparison, analysis, report, presentation video | ✅ Done |

---

## Phase 1 — Traffic Simulation (`src/simulation/traffic_env.py`)

Discrete-time simulation of one intersection, two roads (`TrafficEnv`).

**Inputs (config):**
- `arrival_rate_1`, `arrival_rate_2` — average cars/tick arriving on each road (Poisson process)
- `departure_rate` — max cars that can leave per tick while green
- `clearance_time` — optional all-red ticks when switching (default 0)

**Each `step(green_road)` does, in order:** new arrivals → mark newly-stopped cars → discharge cars from the green road.

**`get_metrics()` returns** the three raw quantities the cost function needs: average waiting time (W), average queue length (Q), number of stops (S).

A naive **fixed-timer baseline** (switch every N ticks regardless of queue length) is used throughout as the point of comparison — it cannot react to real traffic conditions, which motivates Phase 2.

---

## Phase 2 — Fuzzy Controller (`src/fuzzy/fuzzy_controller.py`)

**Inputs (2):** queue length of road 1, queue length of road 2 — each split into 3 fuzzy sets: **Low, Medium, High**.

**Output (1):** green time for road 1 — 3 fuzzy sets: **Short, Medium, Long**. Road 2's green time is derived as `cycle_time - green_time_1` (clipped), since the spec defines only one fuzzy output.

**Rule base:** 9 rules (3×3, all input combinations), each with a tunable **weight** in [0, 1].

**The 4-step Mamdani pipeline, run inside `compute_green_time()`:**

1. **Fuzzification** — each queue length is converted into a membership degree in Low/Medium/High via **triangular membership functions**, each defined by 3 points `(a, b, c)`.
2. **Rule evaluation** — each rule's firing strength = `min(degree_1, degree_2) × rule_weight` (fuzzy AND = minimum).
3. **Aggregation** — for each output set (Short/Medium/Long), take the `max` firing strength across all rules pointing to it (fuzzy OR = union).
4. **Defuzzification (centroid / center of gravity)** — the aggregated output shape is converted to one crisp number by dividing its moment by its area:

```
y* = Σ(xᵢ · μ(xᵢ)) / Σ(μ(xᵢ))
```

computed numerically by sampling the output universe (101 points by default).

**The 27 tunable parameters** (what Phase 3 searches over):

| Count | What |
|---|---|
| 9 | Shape of 3 input membership functions (Low, Medium, High) × 3 points (a,b,c) |
| 9 | Shape of 3 output membership functions (Short, Medium, Long) × 3 points (a,b,c) |
| 9 | Weight of each of the 9 rules |
| **27** | **Total** |

`min_green_time`, `max_green_time`, `cycle_time` are fixed constants, **not** part of these 27 — PSO/ACO can never touch them.

**Cost function** (`src/cost_function.py`): `C = α·W + β·Q + γ·S` (defaults `α=1.0, β=1.0, γ=0.1`), plus `evaluate_controller()` — runs a full simulation with realistic block-based light switching (holds each road green for its computed duration, not flickering every tick) and returns the cost.

---

## Phase 3 — Optimization (`src/optimization/pso.py`, `aco.py`)

Both hand-picked (Phase 2) parameters aren't robust — the fixed-timer baseline can actually beat default fuzzy under light, balanced traffic. Phase 3 searches the 27-dimensional space automatically instead of guessing.

Both algorithms are **black-box optimizers** — neither contains any fuzzy-logic code. They only call:
```python
controller.get_param_bounds()        # search space limits
controller.get_default_vector()      # known-good starting point
controller.set_params_from_vector(x) # try a candidate
evaluate_controller(...)             # score it (minimize)
```

### PSO — Particle Swarm Optimization
A swarm of particles (candidate 27-vectors) move through the search space. Each particle has a velocity, blended from three pulls each iteration:
```
velocity = w·(previous velocity)                    # inertia
         + c1·rand·(personal_best − position)        # pulled to its own best
         + c2·rand·(global_best − position)           # pulled to swarm's best
```
Then `position += velocity`, followed by **repair** (clip to bounds, sort each `(a,b,c)` triangle ascending so it stays a valid membership function).

### ACO — Discretized Ant System (classic pheromone-trail ACO)
Classic ACO (Dorigo, 1992) is defined over discrete choices: a finite set of options per step, a real pheromone value per option, probabilistic selection weighted by pheromone, and an evaporate-then-deposit update rule. Our search space is continuous (27 real numbers), so this module **manufactures a discrete graph** to run the real algorithm on:

1. **Discretize** — each of the 27 dimensions is sliced into `num_bins` (default 20) evenly-spaced levels between its lower and upper bound. "Build a solution" becomes "for each dimension, pick one of `num_bins` levels" — a genuine discrete choice.
2. **Pheromone table** — a real matrix `tau[dimension, bin]`, one value per (dimension, bin) cell. This table *is* the algorithm's memory (replacing what an archive did in the earlier continuous version).
3. **Construct an ant** — for each dimension independently, pick a bin via roulette-wheel selection weighted by `tau[dim, :] ** alpha` (higher pheromone = more likely to be picked), decode the chosen bin to its center value, repair (clip + sort triangles).
4. **Update pheromone** each iteration, exactly Dorigo's original Ant System rule:

```
τᵢⱼ = (1 − ρ)·τᵢⱼ + Δτᵢⱼ
```

   - **Evaporation**: every cell decays by `(1 − ρ)`, floored at `tau_min` so no bin's probability ever collapses to exactly zero (Max-Min-style safeguard).
   - **Deposit**: each ant reinforces the bins it used by `Δτ = Q / cost` — better (lower-cost) solutions deposit more pheromone.
   - **Elitist deposit**: the best-known solution gets extra reinforcement on top of its own deposit every iteration, so evaporation can never erase the best trail found so far.

**Trade-off vs. the earlier continuous approach**: parameters now snap to one of `num_bins` bin centers rather than varying continuously, so resolution is capped by `num_bins`. This is the deliberate cost of using the textbook pheromone-trail formulation faithfully, rather than sidestepping discretization with a continuous-domain variant.

### Real numbers (small pipeline check, 25 iterations, single scenario)

| Controller | Cost | vs. default |
|---|---|---|
| Default (hand-picked) | 40.28 | — |
| PSO-tuned | 34.41 | −14.6% |
| ACO-tuned | 36.24 | −10.0% |

*(These ACO numbers are from the earlier continuous-domain (ACO_R) version, before the switch to discretized classic Ant System described above. Re-run `experiments/compare_optimizers.py` for up-to-date numbers from the current ACO implementation — see Phase 4.)*

---

## Phase 4 — Comparison & Analysis (`experiments/compare_optimizers.py`)

Both algorithms are evaluated against a **robust multi-scenario fitness**: 4 traffic patterns (moderate/heavy × symmetric/asymmetric) × several seeds, averaged — so the tuned controller generalizes rather than overfitting to one arrival sequence.

**Four comparison criteria produced:**

1. **Final cost-function value** — best cost each algorithm reaches, vs. the hand-picked baseline.
2. **Convergence speed** — best-cost-so-far per iteration, plotted (`results/plots/pso_vs_aco_convergence.png`).
3. **Stability across runs** — each algorithm re-run with several independent seeds; mean ± std of final cost and of the parameter vector itself. Low spread = reliable algorithm regardless of random initialization.
4. **Effect on real controller performance** — the tuned controller's cost broken down per individual scenario (not just the aggregate the optimizer minimized), so a good average can't hide a regression on one traffic pattern.

Outputs: `results/plots/pso_vs_aco_convergence.png`, `results/tables/pso_vs_aco_summary.md` (+ `.csv`), `results/tables/pso_vs_aco_result_params.md` (+ `.csv`).

---

## Structure

```
smart-traffic-light-controller/
├── data/simulation_logs/            generated run data
├── src/
│   ├── simulation/                  Phase 1 — traffic simulation       ✅
│   ├── fuzzy/                       Phase 2 — fuzzy controller         ✅
│   ├── optimization/                Phase 3 — pso.py, aco.py          ✅
│   └── cost_function.py             Phase 2 — C = αW + βQ + γS         ✅
├── experiments/
│   ├── test_traffic_env.py          Phase 1 smoke test
│   ├── test_fuzzy_controller.py     Phase 2 smoke test
│   ├── test_pso_smoke.py            Phase 3 PSO smoke test
│   ├── test_aco_smoke.py            Phase 3 ACO smoke test
│   ├── run_pso_optimization.py      Phase 3 full PSO run
│   ├── run_aco_optimization.py      Phase 3 full ACO run
│   └── compare_optimizers.py        Phase 4 comparison (plots + tables)
├── results/
│   ├── plots/                       convergence charts
│   └── tables/                      comparison + parameter tables
├── notebooks/                       optional exploration
└── report/                          final written report
```

Each `src/` submodule has its own README with full implementation details.

## Setup & Running

```bash
pip install -r requirements.txt

# Smoke tests (fast, sanity checks only)
python experiments/test_traffic_env.py
python experiments/test_fuzzy_controller.py
python experiments/test_pso_smoke.py
python experiments/test_aco_smoke.py

# Full Phase 3 runs (slow — see each script's docstring for expected runtime)
python experiments/run_pso_optimization.py
python experiments/run_aco_optimization.py

# Phase 4 comparison (produces plots + tables in results/)
python experiments/compare_optimizers.py
```

## Authors

Course project — Computational Intelligence, Semester project.