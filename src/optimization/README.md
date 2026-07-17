# Phase 3 — Optimization (PSO & ACO)

**Code:** `src/optimization/pso.py`, `src/optimization/aco.py`
**Depends on:** `src/fuzzy/fuzzy_controller.py` (Phase 2), `src/cost_function.py` (Phase 2)
**Status:** ✅ Both algorithms implemented, verified against Phase 2, and consistent with each other's interface

## What this is, in plain words

Phase 2 gave us a fuzzy controller with **27 hand-picked numbers** (membership function shapes + rule weights) that decide how long each road's green light stays on. Those hand-picked numbers aren't robust — the Phase 2 README shows the fixed-timer baseline actually *beats* the default fuzzy controller under light, balanced traffic.

This phase's job: **stop guessing the 27 numbers by hand, and search for better ones automatically.** Two different search algorithms are provided:

- **PSO (Particle Swarm Optimization)** — a "swarm" of candidate solutions moves through the 27-dimensional search space, pulled toward the best solution any individual has personally found and the best solution the whole swarm has found.
- **ACO (Ant Colony Optimization, discretized classic Ant System)** — each of the 27 dimensions is sliced into discrete bins; a real pheromone table over (dimension, bin) pairs guides where new candidates are sampled, updated every iteration by evaporation + deposit, exactly Dorigo's original Ant System rule.

Both algorithms are **generic black-box optimizers** — neither file contains a single line of fuzzy-logic code. They only ever touch the fuzzy controller through the contract already exposed by Phase 2:

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

### How it works, step by step

Think of a flock of birds (particles) searching for the best feeding ground. Each particle carries a position (a candidate 27-number vector) and a velocity (its current direction/speed of movement across all 27 dimensions at once).

**1. Initialization** (`_initialize_population`): `num_particles` particles are created at random positions within bounds, each with a random initial velocity, and evaluated once. By default (`seed_with_default_vector=False`) every particle starts from a random point — the Phase 2 hand-picked baseline is **not** automatically included in the swarm unless this flag is explicitly turned on.

**2. Every iteration, for every particle:**
```
velocity = w · (previous velocity)                       # inertia: keep drifting the same way
         + c1 · rand · (personal_best − current position)  # pulled toward its own best-ever spot
         + c2 · rand · (global_best − current position)    # pulled toward the swarm's best-ever spot
position = position + velocity
```
followed by **repair** (clip to bounds, sort each triangle so `a ≤ b ≤ c` still holds), then re-evaluation.

**3. Bookkeeping**: if a particle's new fitness beats its own personal best, personal best updates; if it beats the swarm's global best, global best updates too.

**4. Inertia decay** (optional, via `w_min`): `w` linearly shrinks from `w` to `w_min` over the run, so the swarm explores broadly early on and settles down (exploits) later.

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
| `seed_with_default_vector` | **Default `False`.** If `True`, one particle starts exactly at the Phase 2 hand-picked baseline instead of a random point. With the default `False`, PSO always starts from a fully random swarm — it can, in principle, do worse than the hand-picked baseline on a given run, so always compare its result against the baseline cost explicitly (Phase 4 does this). |
| `random_seed` | Reproducibility. |

### What you get back

- `best_position` — the winning 27-number vector.
- `best_cost` — its cost (lower is better).
- `history` — best-cost-so-far after each iteration, for a convergence plot. Monotonically non-increasing by construction (global best only updates on strict improvement).

---

## `aco.py`

Uses the exact same `FitnessFn` contract, `get_param_bounds()`/`get_default_vector()`/`set_params_from_vector()` calls, and `_repair()` logic as `pso.py` — it even imports `FitnessFn`, `make_single_scenario_fitness`, and `make_multi_scenario_fitness` directly from `pso.py` rather than redefining them, so there is exactly one definition of "how do we score a controller" in the whole project.

### Why discretized, classic Ant System — not a continuous-domain variant

Classic ACO (Dorigo, 1992) is defined over **discrete choices**: a finite set of options at each decision point, a real pheromone value per option, probabilistic selection weighted by pheromone, and an evaporate-then-deposit update rule — the textbook formula:

```
τᵢⱼ = (1 − ρ)·τᵢⱼ + Δτᵢⱼ
```

Our search space is a continuous 27-dimensional real vector, so there's no native graph to run this on. `aco.py` **manufactures one**: each of the 27 dimensions is sliced into `num_bins` (default 20) evenly-spaced discrete levels between its lower and upper bound. "Build a candidate solution" becomes "for each dimension, pick one of `num_bins` levels" — a genuine discrete choice, with a genuine pheromone table and genuine evaporation/deposit, rather than sidestepping discretization with a continuous-domain variant.

**Trade-off**: parameters snap to bin centers instead of varying continuously, so resolution is capped by `num_bins`. This is the deliberate cost of applying the textbook algorithm faithfully.

### Running ACO

```python
from optimization.aco import ACOOptimizer

aco = ACOOptimizer(
    controller,
    fitness_fn,
    archive_size=20,   # size of the initial random-sampling batch that primes the pheromone table
    num_ants=10,       # ants constructed per iteration
    max_iter=100,
    q=0.3,             # Ant-System deposit constant Q, used as delta_tau = Q / cost
    xi=0.85,           # reused as evaporation rate rho (clipped to [0.01, 0.99])
    random_seed=7,
)
best_position, best_cost, history = aco.optimize()
```

**Naming note:** to stay a drop-in replacement for the earlier continuous-domain version, this class keeps the old parameter names (`archive_size`, `q`, `xi`) but repurposes their meaning — see the table below. New, ACO-specific knobs (`num_bins`, `alpha`, `tau0`, `tau_min`, `elitist_weight`) are added with sensible defaults.

### How it works, step by step

There's no "movement" here at all — every ant is built fresh each iteration by walking a pheromone table:

**1. Discretize.** Each of the 27 dimensions gets its own row of `num_bins` bin centers, evenly spaced between that dimension's lower and upper bound (`bin_centers[dim, bin]`).

**2. Pheromone table.** A real matrix `tau[dimension, bin]` — this *is* the algorithm's memory. All cells start at `tau0`.

**3. Construct an ant** (`_construct_ant`): for each of the 27 dimensions independently, pick one bin via roulette-wheel selection weighted by `tau[dim, :] ** alpha` — higher pheromone in a cell means a higher chance that bin gets picked. Decode the chosen bin to its center value, then repair (clip + sort triangles).

**4. Initial priming batch.** Before the main loop, `archive_size` ants are built against a still-uniform pheromone table (equivalent to random sampling, since uniform pheromone means uniform selection probability), evaluated, checked against the global best, and used for one pheromone update — the discretized analogue of randomly initializing an archive/swarm. The best cost after this priming batch is `history[0]`.

**5. Every iteration:**
   - Build `num_ants` new ants from the *current* pheromone table.
   - Update the running global best if any ant beats it.
   - **Update pheromone**, exactly Dorigo's Ant System rule:
     - **Evaporate**: `tau *= (1 − rho)`, floored at `tau_min` so no bin's selection probability ever collapses to exactly zero (a Max-Min-style safeguard against premature convergence).
     - **Deposit**: each ant reinforces the bins it used by `Δτ = Q / cost` — lower-cost (better) solutions deposit more pheromone.
     - **Elitist deposit**: the best solution found so far *anywhere in the run* gets an extra reinforcement (`elitist_weight × Q / best_cost`) on top of its own deposit every iteration, so evaporation can never erase the best-known trail.

`history` is **monotonically non-increasing by construction** — the global best is tracked independently of the pheromone table, so evaporation can lower selection probabilities but can never make the recorded best-known cost worse.

### Parameters, explained

| Parameter | What it controls |
|---|---|
| `archive_size` | Size of the initial random-sampling batch used to prime the pheromone table before the main loop starts. |
| `num_ants` | New candidate solutions constructed per iteration. |
| `max_iter` | Number of construct → evaluate → update-pheromone cycles. |
| `q` | Ant-System deposit constant `Q` in `Δτ = Q / cost`. Larger `Q` means every ant's deposit (relative to evaporation) is stronger. |
| `xi` | Reused as pheromone evaporation rate `ρ`, clipped to `[0.01, 0.99]`. Higher = faster forgetting of old pheromone → faster but riskier (more premature) convergence. |
| `num_bins` | Number of discrete levels each of the 27 dimensions is sliced into. More bins = finer resolution but a bigger table to learn (slower convergence for the same ant budget). Default 20. |
| `alpha` | Pheromone-influence exponent — selection probability for a bin ∝ `tau ** alpha`. `alpha=1.0` (default) is standard Ant System; higher values make ants follow the strongest trails more greedily. |
| `tau0` | Initial pheromone value in every cell. |
| `tau_min` | Hard floor on pheromone after evaporation, keeping exploration alive (Max-Min Ant System style). |
| `elitist_weight` | Extra deposit multiplier applied to the global-best solution's bins every update, on top of its own regular deposit. |
| `seed_with_default_vector` | **Default `False`.** If `True`, the pheromone table is pre-boosted (×5, via `_bias_toward_default_vector`) at the bin nearest to the Phase 2 baseline in every dimension — a soft bias, not a guarantee it's ever sampled, since selection is still probabilistic. |
| `random_seed` | Reproducibility. |

### What you get back

Same shape as PSO: `(best_position, best_cost, history)`.

---

## PSO vs. ACO — when to use which

| | PSO | ACO (discretized Ant System) |
|---|---|---|
| Search mechanism | Velocity-driven movement of particles through continuous space | Discrete bin selection guided by a pheromone table, updated by evaporation + deposit |
| Resolution | Full continuous precision | Capped by `num_bins` — parameters snap to bin centers |
| Tends to | Converge faster on smooth, single-peaked landscapes | Textbook-faithful pheromone dynamics, at the cost of discretization resolution |
| Key exploration knob | `w` (inertia) | `num_bins`, `alpha` |
| Key convergence knob | `w_min` (inertia decay) | `xi` → `ρ` (evaporation rate) |

The project spec asks for both so they can be compared against each other and against the Phase 2 hand-tuned baseline — that comparison is what Phase 4 is for. Neither is "more correct"; they're two different search strategies over the identical 27-dimensional space and identical cost function, with ACO deliberately paying a discretization cost in exchange for being the genuine textbook algorithm rather than a continuous-domain workaround.

---

## Compatibility with Phase 2 — verification notes

- ✅ Vector layout (`params_to_vector`/`vector_to_params`) matches what both optimizers assume, dimension-for-dimension.
- ✅ `get_param_bounds()` output is respected — every optimized vector stays within bounds after repair.
- ✅ Triangle ordering (`a ≤ b ≤ c`) is enforced correctly by `_repair()` in both files.
- ✅ `min_green_time` / `max_green_time` / `cycle_time` correctly stay outside the tuned vector, matching the Phase 2 invariant.
- ✅ Both `PSOOptimizer.optimize()` and `ACOOptimizer.optimize()` run a full loop against real `TrafficEnv` + `FuzzyController` + `evaluate_controller()` without errors, and produce monotonically improving best-cost histories.
- ⚠️ **Note on `seed_with_default_vector`:** both classes default this to `False`. Neither optimizer is automatically guaranteed to match or beat the Phase 2 baseline on a given run unless this flag is turned on — always report the baseline cost alongside PSO/ACO results (Phase 4's `compare_optimizers.py` does this).

## What Phase 4 uses this for

`experiments/run_pso_optimization.py`, `experiments/run_aco_optimization.py`, and `experiments/compare_optimizers.py` all consume this module's public interface (`PSOOptimizer`, `ACOOptimizer`, `make_multi_scenario_fitness`) unchanged — see the top-level project README and each script's own docstring for how the full-scale, multi-scenario comparison is run.