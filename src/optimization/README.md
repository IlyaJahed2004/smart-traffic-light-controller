# Phase 3 — Optimization (PSO & ACO)

**Code:** `src/optimization/pso.py`, `src/optimization/aco.py`
**Depends on:** `src/fuzzy/fuzzy_controller.py` (Phase 2), `src/cost_function.py` (Phase 2)
**Status:** ✅ Verified compatible with the fixed Phase 2 code — one stale docstring bug found and fixed, otherwise unchanged

## What this is, in plain words

Phase 2 gave us a fuzzy controller with **27 hand-picked numbers** (membership function shapes + rule weights) that decide how long each road's green light stays on. Those hand-picked numbers aren't robust — the Phase 2 README shows the fixed-timer baseline actually *beats* the default fuzzy controller under light, balanced traffic.

This phase's job: **stop guessing the 27 numbers by hand, and search for better ones automatically.** Two different search algorithms are provided:

- **PSO (Particle Swarm Optimization)** — a "swarm" of candidate solutions moves through the 27-dimensional search space, pulled toward the best solution any individual has personally found and the best solution the whole swarm has found.
- **ACO_R (Ant Colony Optimization for continuous domains)** — a pool ("archive") of the best solutions found so far acts as a set of landing zones; new candidates are sampled near existing good solutions, weighted toward the best-ranked ones.

Both algorithms are **generic black-box optimizers** — neither file contains a single line of fuzzy-logic code. They only ever touch the fuzzy controller through four methods it already exposes from Phase 2:

```python
controller.get_param_bounds()        # search space limits
controller.get_default_vector()      # a known-good starting point
controller.set_params_from_vector(x) # try a candidate
# ...then evaluate_controller() from cost_function.py scores it
```

This is exactly the contract described in `src/fuzzy/README.md` — Phase 3 was written by someone who never needed to understand triangular membership functions or centroid defuzzification.

---

## The shared contract with Phase 2

Every candidate solution is a **flat numpy array of 27 numbers**, laid out exactly as `FuzzyController.params_to_vector()` produces it:

| Indices | Meaning |
|---|---|
| `[0:3]` | Queue MF "Low" — `(a, b, c)` |
| `[3:6]` | Queue MF "Medium" — `(a, b, c)` |
| `[6:9]` | Queue MF "High" — `(a, b, c)` |
| `[9:12]` | Green MF "Short" — `(a, b, c)` |
| `[12:15]` | Green MF "Medium" — `(a, b, c)` |
| `[15:18]` | Green MF "Long" — `(a, b, c)` |
| `[18:27]` | 9 rule weights, one per rule in `FuzzyController.RULES` |

`min_green_time`, `max_green_time`, and `cycle_time` are **not** part of this vector — they're fixed constants set once when `FuzzyController` is constructed (see Phase 2 README), so neither PSO nor ACO can ever touch them.

### The repair step (why every candidate is valid)

Both optimizers generate raw candidates that could, in principle, violate two things:
1. Go outside the bounds `get_param_bounds()` returns.
2. Produce a triangle `(a, b, c)` where `a > b` or `b > c` — not a valid triangular membership function, since `_triangular_membership()` assumes ascending order.

Both files fix this with the same `_repair()` step: **clip to bounds, then sort each of the 6 `(a, b, c)` triplets ascending.** This is safe specifically because `a`, `b`, `c` share identical bounds within each triangle (e.g. all three of a queue MF's points are bounded by `[0, q_max]`) — so sorting can never push a value outside the range it was already clipped to. Rule weights (indices `18:27`) only need the bounds clip, since there's no ordering constraint on them.

*(This was flagged as a risk before Phase 3 was written — it's already handled correctly in both files, verified by inspecting real optimizer output: every returned vector has `a ≤ b ≤ c` in all six triangles.)*

---

## `pso.py`

### Building a fitness function

Before you can optimize anything, you need a function that takes a *configured* controller and returns a single cost number to minimize. Two builders are provided:

```python
from optimization.pso import make_single_scenario_fitness, make_multi_scenario_fitness
from simulation.traffic_env import TrafficEnv

# Single traffic scenario
fitness_fn = make_single_scenario_fitness(
    env_factory=lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2,
                                    departure_rate=1.0, seed=42),
    num_steps=1000,
)

# Averaged across several scenarios (recommended — see "Robustness" below)
fitness_fn = make_multi_scenario_fitness(
    env_factories=[
        lambda: TrafficEnv(arrival_rate_1=0.3, arrival_rate_2=0.3, departure_rate=1.0, seed=42),
        lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0, seed=42),
        lambda: TrafficEnv(arrival_rate_1=0.6, arrival_rate_2=0.2, departure_rate=1.0, seed=42),
        lambda: TrafficEnv(arrival_rate_1=0.5, arrival_rate_2=0.5, departure_rate=1.0, seed=42),
    ],
    num_steps=1000,
)
```

Both accept `alpha`, `beta`, `gamma` as extra keyword arguments, forwarded straight to `compute_cost()` in `cost_function.py` (defaults: `1.0, 1.0, 0.1`). Nothing else is accepted — `evaluate_controller()` always does realistic block-based light switching internally (a Phase 2 fix), so there's no per-tick timing knob to configure here.

### Running PSO

```python
from fuzzy.fuzzy_controller import FuzzyController
from optimization.pso import PSOOptimizer

controller = FuzzyController()

pso = PSOOptimizer(
    controller,
    fitness_fn,
    num_particles=30,
    max_iter=100,
    w=0.7, c1=1.5, c2=1.5,
    random_seed=7,
)
best_position, best_cost, history = pso.optimize()

# Load the winning parameters back into a controller to actually use it
controller.set_params_from_vector(best_position)
```

### Parameters, explained

| Parameter | What it controls |
|---|---|
| `num_particles` | Swarm size — how many candidate solutions explore in parallel each iteration. |
| `max_iter` | Number of update rounds. |
| `w` | Inertia — how much of a particle's previous velocity carries over. Higher = more exploration, slower to settle. |
| `c1` | Cognitive coefficient — how strongly a particle is pulled toward *its own* best-ever position. |
| `c2` | Social coefficient — how strongly a particle is pulled toward the *swarm's* best-ever position. |
| `w_min` | If set, inertia decays linearly from `w` to `w_min` over the run (start exploratory, end exploitative). Leave `None` for constant inertia. |
| `v_max_fraction` | Caps per-dimension velocity at this fraction of that dimension's range, so particles can't leap across the whole search space in one step. |
| `seed_with_default_vector` | If `True` (default), one particle starts exactly at the Phase 2 hand-picked baseline instead of a random point — guarantees PSO can never do *worse* than the default, since it's already in the swarm. |
| `random_seed` | Reproducibility. |

### What you get back

- `best_position` — the winning 27-number vector.
- `best_cost` — its cost (lower is better).
- `history` — best-cost-so-far after each iteration, for a convergence plot.

---

## `aco.py`

Uses the exact same `FitnessFn` contract, `get_param_bounds()`/`get_default_vector()`/`set_params_from_vector()` calls, and `_repair()` logic as `pso.py` — it even imports `FitnessFn`, `make_single_scenario_fitness`, and `make_multi_scenario_fitness` directly from `pso.py` rather than redefining them, so there is exactly one definition of "how do we score a controller" in the whole project.

### Running ACO

```python
from optimization.aco import ACOOptimizer

aco = ACOOptimizer(
    controller,
    fitness_fn,
    archive_size=20,
    num_ants=10,
    max_iter=100,
    q=0.3, xi=0.85,
    random_seed=7,
)
best_position, best_cost, history = aco.optimize()
```

### Parameters, explained

| Parameter | What it controls |
|---|---|
| `archive_size` (K) | Number of best-so-far solutions kept as "landing zones" for new ants. ACO_R's equivalent of PSO's swarm size. Must be ≥ 2. |
| `num_ants` | New candidate solutions constructed per iteration. |
| `max_iter` | Number of construct → evaluate → merge → truncate cycles. |
| `q` | Locality. Small `q` concentrates new ants near the *top-ranked* archive members (exploitation); larger `q` spreads attention more evenly across the whole archive (exploration). |
| `xi` | Convergence speed. Larger `xi` shrinks the sampling spread faster as the archive agrees on a value — similar in spirit to PSO's inertia decay. |
| `seed_with_default_vector` | Same idea as PSO — one archive slot starts at the Phase 2 baseline. |
| `random_seed` | Reproducibility. |

### How it works, one level deeper

Unlike PSO (particles moving with velocity), ACO_R has no notion of "movement" — every ant is built from scratch each iteration:

1. For each of the 27 dimensions independently, pick one archive member as a "guide," with better-ranked members more likely to be picked (Gaussian-weighted by rank — controlled by `q`).
2. Sample a new value for that dimension from a Gaussian centered on the guide's value, with spread proportional to how much the archive already disagrees on that dimension (scaled by `xi`).
3. Repair (clip + sort triangles), evaluate, and merge all new ants into the archive.
4. Keep only the best `archive_size` solutions overall — this elitist truncation is ACO_R's version of a "pheromone update": good solutions persist and keep attracting ants, bad ones are discarded.

`history` in ACO's output is **monotonically non-increasing by construction** — elitist truncation means the best-known solution can never get worse from one iteration to the next, only stay the same or improve. PSO's global best has the same property; the difference is in *how* new candidates are generated.

### What you get back

Same shape as PSO: `(best_position, best_cost, history)`.

---

## PSO vs. ACO — when to use which

| | PSO | ACO_R |
|---|---|---|
| Search mechanism | Velocity-driven movement of particles | Fresh sampling around an archive of good solutions |
| Tends to | Converge faster on smooth, single-peaked landscapes | Handle rugged/multi-modal landscapes a bit more gracefully, at the cost of more evaluations to build a good archive |
| Key exploration knob | `w` (inertia) | `q` (locality) |
| Key convergence knob | `w_min` (inertia decay) | `xi` (spread decay) |

The project spec asks for both so they can be compared against each other and against the Phase 2 hand-tuned baseline — that comparison is what Phase 4 is for. Neither is "more correct"; they're two different search strategies over the identical 27-dimensional space and identical cost function.

---

## Worked example (real output, not illustrative)

Run on `arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0`, `num_steps=1000`, `random_seed=7`:

```python
from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import evaluate_controller
from optimization.pso import PSOOptimizer, make_single_scenario_fitness
from optimization.aco import ACOOptimizer

env_factory = lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0, seed=42)
fitness_fn = make_single_scenario_fitness(env_factory, num_steps=1000)

default_cost = evaluate_controller(FuzzyController(), env_factory, num_steps=1000)

pso = PSOOptimizer(FuzzyController(), fitness_fn, num_particles=15, max_iter=25, random_seed=7)
_, pso_cost, pso_history = pso.optimize()

aco = ACOOptimizer(FuzzyController(), fitness_fn, archive_size=15, num_ants=8, max_iter=25, random_seed=7)
_, aco_cost, aco_history = aco.optimize()
```

Actual results from this run:

| Controller | Cost | Improvement vs. default |
|---|---|---|
| Default (Phase 2 hand-picked) | 40.28 | — |
| PSO-tuned (15 particles, 25 iters) | 34.41 | −14.6% |
| ACO-tuned (archive 15, 8 ants/iter, 25 iters) | 36.24 | −10.0% |

Convergence (best cost so far, every 5th iteration):

```
PSO: [40.28, 37.37, 36.78, 34.74, 34.62, 34.41]
ACO: [40.28, 40.28, 36.78, 36.78, 36.24, 36.24]
```

Both algorithms improve on the hand-picked baseline with a fairly small budget (25 iterations, a few hundred simulation runs total). This is a single scenario at a small iteration count purely to demonstrate the pipeline works end-to-end — it is **not** the final Phase 3 result. Real experiments should use `make_multi_scenario_fitness` across the four scenarios from the Phase 2 README, larger `num_particles`/`archive_size`, and more iterations, per the recommendation already in `src/fuzzy/README.md`.

---

## Compatibility with Phase 2 — verification notes

Before being handed off, both files were run end-to-end against the actual (fixed) `fuzzy_controller.py` and `cost_function.py`, not just read:

- ✅ Vector layout (`params_to_vector`/`vector_to_params`) matches what both optimizers assume, dimension-for-dimension.
- ✅ `get_param_bounds()` output is respected — every optimized vector stays within bounds after repair.
- ✅ Triangle ordering (`a ≤ b ≤ c`) is enforced correctly by `_repair()` in both files — checked directly on real optimizer output, not just by reading the code.
- ✅ `min_green_time` / `max_green_time` / `cycle_time` correctly stay outside the tuned vector, matching the Phase 2 invariant.
- ✅ Both `PSOOptimizer.optimize()` and `ACOOptimizer.optimize()` run a full loop against real `TrafficEnv` + `FuzzyController` + `evaluate_controller()` without errors, and produce monotonically improving best-cost histories.
- 🔧 **One bug found and fixed:** `pso.py`'s `make_single_scenario_fitness()` docstring listed `decision_interval` as a valid `cost_kwargs` option. That was a leftover from before the Phase 2 block-switching fix — `evaluate_controller()` no longer accepts it, and calling it as documented raised `TypeError`. The docstring now correctly states only `alpha`/`beta`/`gamma` are accepted. No functional code was changed — `aco.py` required no changes at all.

## What's not done yet / open for Phase 4

- `run_pso_optimization.py`, `run_aco_optimization.py`, and `compare_algorithms.py` (referenced in the project file tree) are not covered by this README — they weren't reviewed as part of this pass.
- No convergence plots yet (the `history` list from both optimizers is ready to feed straight into `matplotlib`).
- Multi-scenario fitness (`make_multi_scenario_fitness`) is implemented and available but not yet the default in any run script — worth confirming both teammates are using it before locking in final results, per the robustness discussion in `src/fuzzy/README.md`.
- No head-to-head PSO vs. ACO comparison at full scale (large `num_particles`/`archive_size`, many iterations, all four traffic scenarios) has been run yet — the table above is a small-scale pipeline check only.