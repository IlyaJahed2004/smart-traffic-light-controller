# Optimizers: PSO and ACO

This document explains how `optimization/pso.py` and `optimization/aco.py`
work, how they relate to each other, and how to actually call them from
your own code. Both optimizers do the same job — tune the
`FuzzyController`'s 27-parameter vector to minimize traffic cost — through
two different search strategies. They are **interchangeable**: anything
that works with one works with the other, because both are built against
the exact same contract.

---

## 1. The shared contract

Neither optimizer contains any fuzzy-logic or traffic-simulation code.
They only ever touch the controller and simulation through four calls:

```python
controller.get_param_bounds()       # -> (lower: np.ndarray, upper: np.ndarray), both shape (27,)
controller.get_default_vector()     # -> np.ndarray, shape (27,) — the hand-tuned baseline
controller.set_params_from_vector(x)  # apply a candidate vector to the controller
evaluate_controller(controller, env_factory, num_steps, **cost_kwargs)  # -> float cost
```

This is why the module docstrings call the controller/simulation a
"black box" — as long as something implements this interface, both
optimizers can tune it, and neither needs to know anything about fuzzy
membership functions or traffic dynamics.

### The 27-dimensional vector layout

Both optimizers' *repair* step (see §4) depends on this layout, defined
authoritatively in `FuzzyController.params_to_vector`:

| Indices | Meaning |
|---|---|
| `[0:3]`   | `mf_queue` "Low"    triangle (a, b, c) |
| `[3:6]`   | `mf_queue` "Medium" triangle (a, b, c) |
| `[6:9]`   | `mf_queue` "High"   triangle (a, b, c) |
| `[9:12]`  | `mf_green` "Short"  triangle (a, b, c) |
| `[12:15]` | `mf_green` "Medium" triangle (a, b, c) |
| `[15:18]` | `mf_green` "Long"   triangle (a, b, c) |
| `[18:27]` | `rule_weights[0..8]` |

The first 18 entries are six membership-function *triangles* (3 queue
sets + 3 green-time sets), each needing `a <= b <= c` to be a valid
triangle. The last 9 are rule weights, each independently clipped to
`[0, 1]` — no ordering constraint between them.

Within each triangle, `get_param_bounds()` returns **identical**
lower/upper bounds for `a`, `b`, and `c` (e.g. all three share
`[0, q_max]`). That's what makes the repair trick in §4 safe.

### `FitnessFn`: how "goodness" is defined

```python
FitnessFn = Callable[[object], float]
```

A fitness function takes an **already-configured** controller (i.e.
`set_params_from_vector` has already been called on it) and returns a
scalar cost to **minimize**. Both optimizers are written against this
exact signature and don't care how the cost is computed internally.

`pso.py` provides two factory functions that build a `FitnessFn` for
you — `aco.py` re-exports and reuses these rather than redefining them,
so there is only one definition of "how do we score a controller" for
the whole project:

```python
from optimization.pso import make_single_scenario_fitness, make_multi_scenario_fitness

# Single traffic scenario, evaluated once per candidate:
fitness_fn = make_single_scenario_fitness(
    env_factory=lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2,
                                    departure_rate=1.0, seed=42),
    num_steps=500,
)

# Several scenarios, cost averaged (or otherwise aggregated) across all of them —
# use this for a controller that isn't overfit to one specific arrival sequence:
fitness_fn = make_multi_scenario_fitness(
    env_factories=[factory1, factory2, factory3, ...],
    num_steps=500,
    aggregate=np.mean,   # or np.max for a worst-case-focused objective, etc.
)
```

**Important:** `env_factory` is a zero-argument *callable that builds* an
environment, not an environment instance. Both optimizers evaluate
hundreds or thousands of candidates, and each evaluation needs a fresh,
zero-state environment — reusing one instance would leak leftover queue
state between evaluations. Always pass a factory (a `lambda` or small
function), never a pre-built `TrafficEnv`.

Any extra keyword arguments (`alpha`, `beta`, `gamma` — the cost-function
weights in `cost_function.py`) get forwarded straight through:

```python
fitness_fn = make_single_scenario_fitness(env_factory, num_steps=500,
                                           alpha=1.0, beta=0.5, gamma=0.2)
```

---

## 2. PSO — Particle Swarm Optimization

**File:** `optimization/pso.py` · **Class:** `PSOOptimizer`

### The idea

A swarm of `num_particles` candidate vectors ("particles") move through
the 27-dimensional search space simultaneously. Each particle has a
position (a candidate vector) and a velocity. On every iteration, each
particle's velocity is nudged toward two things:

- its own **personal best** position ever found (`personal_best_position`) — the "cognitive" pull, weighted by `c1`
- the swarm's **global best** position ever found (`global_best_position`) — the "social" pull, weighted by `c2`

...plus some inertia (`w`) carrying over its previous velocity, and
randomness (`r1`, `r2`, redrawn every step) so the swarm doesn't move in
lockstep. This is the standard global-best PSO update rule.

### Minimal usage

```python
from fuzzy.fuzzy_controller import FuzzyController
from optimization.pso import PSOOptimizer, make_single_scenario_fitness

controller = FuzzyController()
fitness_fn = make_single_scenario_fitness(
    env_factory=lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2,
                                    departure_rate=1.0, seed=42),
    num_steps=500,
)

pso = PSOOptimizer(controller, fitness_fn, num_particles=30, max_iter=100)
best_position, best_cost, history = pso.optimize()

controller.set_params_from_vector(best_position)  # load the tuned result
```

### Constructor parameters

| Parameter | Default | Meaning |
|---|---|---|
| `controller` | — | Used only to read bounds/default vector; its logic is never modified. |
| `fitness_fn` | — | A `FitnessFn` — see §1. |
| `num_particles` | `30` | Swarm size. |
| `max_iter` | `100` | Number of iterations. |
| `w` | `0.7` | Inertia weight — how much of the previous velocity carries over. |
| `c1` | `1.5` | Cognitive coefficient — pull toward the particle's own best. |
| `c2` | `1.5` | Social coefficient — pull toward the swarm's global best. |
| `w_min` | `None` | If set, `w` decays linearly from `w` down to `w_min` over the run (more exploration early, more exploitation late). If `None`, `w` stays constant. |
| `v_max_fraction` | `0.2` | Caps per-dimension velocity at this fraction of that dimension's `(upper - lower)` range, so particles can't routinely overshoot the whole search space in one step. |
| `random_seed` | `None` | Seed for reproducible runs. |
| `seed_with_default_vector` | `False` | If `True`, particle 0 starts exactly at `controller.get_default_vector()` instead of a random position. |

### What `optimize()` does, step by step

1. **`_initialize_population()`** — builds `num_particles` particles at
   random positions (or one at the default vector, if
   `seed_with_default_vector=True`), gives each a small random initial
   velocity, evaluates every particle's fitness once, and sets the
   initial personal/global bests.
2. For each of `max_iter` iterations:
   - Compute the current inertia (`_current_inertia`, linearly decayed
     if `w_min` was set).
   - For every particle: update velocity → update position (`position +
     velocity`, then repaired) → evaluate fitness → update its personal
     best → update the swarm's global best.
   - Append the current global best cost to `history`.
3. Return `(global_best_position, global_best_cost, history)`.

`history` is **monotonically non-increasing** by construction — the
global best is only ever overwritten by something *strictly better*, so
the convergence curve can never regress.

---

## 3. ACO — discretized Ant System

**File:** `optimization/aco.py` · **Class:** `ACOOptimizer`

### The idea, and why it's discretized

"Textbook" ACO (Dorigo's Ant System) is defined over a discrete graph —
a finite set of choices at each step, with a real pheromone value per
choice, probabilistic selection weighted by pheromone, and
evaporation + deposit as the update rule. There's no native graph on a
continuous 27-D vector, so this module *manufactures* one:

- Each of the 27 dimensions is sliced into `num_bins` evenly-spaced
  discrete levels between its lower/upper bound.
- A real pheromone table `tau[dim, bin]` (shape `(27, num_bins)`) stores
  how attractive each bin has proven to be so far — this table **is**
  the algorithm's memory (analogous to PSO's swarm of particles, but a
  totally different data structure).
- Each ant builds one candidate vector by making 27 independent,
  pheromone-weighted random choices (roulette-wheel selection, Dorigo's
  classic transition rule) — one bin per dimension — then decodes those
  bins to real numbers via precomputed bin centers.
- After a batch of ants is built and evaluated, the pheromone table is
  updated: **evaporate** everything by `(1 - rho)`, then each ant
  **deposits** `Q / cost` onto the bins it used (better solutions
  deposit more), plus a bit of extra reinforcement for the best solution
  found so far so it doesn't get evaporated away.

This is a genuine discrete-choice, pheromone-trail algorithm — not
[ACO_R](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Continuous_orbit_ACO)
(a Gaussian-archive variant designed specifically to *avoid*
discretization). The trade-off: parameters snap to one of `num_bins`
levels instead of varying continuously, capping precision. Expect ACO to
plateau at a slightly worse cost than PSO on this kind of smooth
numeric landscape — that's the honest cost of using "real" ACO here
rather than a method built for continuous domains.

### Minimal usage

```python
from fuzzy.fuzzy_controller import FuzzyController
from optimization.aco import ACOOptimizer
from optimization.pso import make_single_scenario_fitness  # shared helper, see §1

controller = FuzzyController()
fitness_fn = make_single_scenario_fitness(
    env_factory=lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2,
                                    departure_rate=1.0, seed=42),
    num_steps=500,
)

aco = ACOOptimizer(controller, fitness_fn, archive_size=20, num_ants=10, max_iter=100)
best_position, best_cost, history = aco.optimize()

controller.set_params_from_vector(best_position)  # load the tuned result
```

### Constructor parameters

> **Naming note:** this class's constructor keeps the same parameter
> *names* as an earlier ACO_R version, so existing call sites
> (`compare_algorithms.py`, etc.) didn't need to change — but `q` and
> `xi` were **re-purposed** to mean something different. Read the
> "was" column carefully if you're carrying over intuition from ACO_R.

| Parameter | Default | Meaning | Was (old ACO_R) |
|---|---|---|---|
| `controller` | — | Same role as in PSO. | same |
| `fitness_fn` | — | Same role as in PSO. | same |
| `archive_size` | `20` | Size of the initial random-sampling batch used to "prime" the pheromone table before the main loop starts. | ACO_R's persistent archive size `K` |
| `num_ants` | `10` | Ants (new candidates) constructed per iteration. | same |
| `max_iter` | `100` | Number of iterations. | same |
| `q` | `0.3` | Ant-System deposit constant `Q`, used as `Δτ = Q / cost`. | ACO_R's locality parameter |
| `xi` | `0.85` | Pheromone evaporation rate `ρ`, internally clipped to `(0.01, 0.99)`. Higher = faster forgetting, faster convergence, more risk of premature convergence. | ACO_R's spread-decay parameter |
| `random_seed` | `None` | Seed for reproducible runs. | same |
| `seed_with_default_vector` | `False` | If `True`, pre-boosts pheromone at the bin nearest to each dimension of `controller.get_default_vector()`. This is a **soft bias**, not a guarantee — unlike PSO's version of this flag, the default vector is never *forced* into the ant population, since ACO selection is inherently probabilistic. | same idea, different mechanism |
| `num_bins` | `20` | Discretization resolution per dimension (new). |
| `alpha` | `1.0` | Pheromone-influence exponent — selection probability ∝ `tau ** alpha`. Higher = more greedy toward strong trails. (new) |
| `tau0` | `1.0` | Initial pheromone value in every cell. (new) |
| `tau_min` | `1e-3` | Floor on pheromone after evaporation, so no bin's probability ever collapses to exactly zero (Max-Min-style safeguard). (new) |
| `elitist_weight` | `2.0` | Extra deposit multiplier applied to the global-best solution's bins on every update, on top of its own regular deposit. (new) |

### What `optimize()` does, step by step

1. **Priming batch:** with a freshly-uniform pheromone table, selection
   is equivalent to random sampling. Build `archive_size` ants this way,
   track the global best, and run one pheromone update — this is the
   discretized analogue of ACO_R's random archive initialization.
2. For each of `max_iter` iterations:
   - Construct `num_ants` new ants (each: 27 pheromone-weighted bin
     choices → decode → repair → evaluate).
   - Update the global best if any ant beat it.
   - Update the pheromone table (evaporate, then deposit — see §3).
   - Append the current global best cost to `history`.
3. Return `(best_position, best_cost, history)`.

Like PSO, the global best is tracked **separately** from the pheromone
table (which keeps evaporating/being reinforced), and is only ever
updated on strict improvement — so `history` is monotonically
non-increasing here too, even though the underlying pheromone table
itself isn't monotonic in any sense.

---

## 4. The repair step (shared logic, separate implementations)

Both `_repair()` methods do the same two things, for the same reason,
even though the candidates arrive at them differently (continuous
velocity step for PSO, independent per-dimension bin draws for ACO):

1. **Clip** every dimension to `[lower_bounds, upper_bounds]`.
2. **Sort** each of the six `(a, b, c)` membership-function triangles
   ascending, so `a <= b <= c` holds.

Step 2 is only safe because of the bound-sharing property mentioned in
§1: within one triangle, `a`, `b`, and `c` all share identical bounds
(and, for ACO, an identical bin grid). Sorting can only reorder values
that were already valid for any position in the triangle — it can never
push a value out of range. Rule weights (indices 18–27) need no repair
beyond the clip, since there's no ordering constraint between them.

---

## 5. Comparing PSO and ACO side-by-side

Because both classes return the exact same `(best_position, best_cost,
history)` tuple from `optimize()`, and both accept the exact same
`FitnessFn`, they're fully drop-in interchangeable in any script:

```python
from optimization.pso import PSOOptimizer, make_multi_scenario_fitness
from optimization.aco import ACOOptimizer

fitness_fn = make_multi_scenario_fitness(env_factories, num_steps=1000)

pso = PSOOptimizer(FuzzyController(), fitness_fn, num_particles=30, max_iter=100, random_seed=7)
aco = ACOOptimizer(FuzzyController(), fitness_fn, archive_size=30, num_ants=15, max_iter=100, random_seed=7)

pso_best_pos, pso_best_cost, pso_history = pso.optimize()
aco_best_pos, aco_best_cost, aco_history = aco.optimize()
```

This is exactly what `experiments/compare_algorithms.py` does, plotting
both `history` arrays on one convergence chart and writing both
`best_position` vectors out as a readable report. See also
`experiments/run_pso_optimization.py` / `run_aco_optimization.py` for
the full Phase 4 versions with progress logging, feasibility checks, a
`QUICK_TEST` mode for fast smoke runs, and final-vector logging to
`results/logs/`.

### Practical differences to expect

| | PSO | ACO |
|---|---|---|
| Search space | Continuous | Discretized (`num_bins` levels/dim) |
| Memory structure | Swarm of particles + velocities | Pheromone table `(27, num_bins)` |
| Precision ceiling | None (continuous) | Capped by `num_bins` |
| Typical convergence | Smooth, tends to reach a lower final cost on smooth landscapes like this one | Can plateau slightly higher due to discretization, but converges via a genuinely different mechanism (useful for comparison writeups) |
| Cost per evaluation batch | `num_particles` evaluations/iteration | `num_ants` evaluations/iteration (plus `archive_size` up front) |

### Tuning tips

- **PSO:** if particles converge too fast and get stuck, increase `c1`
  or lower `w_min`/`w` less aggressively (more exploration).
  `v_max_fraction` matters a lot in high dimensions — too high and
  particles overshoot constantly; too low and they crawl.
- **ACO:** `num_bins` is the main precision/speed trade-off — more bins
  means finer resolution but a larger table to learn with the same ant
  budget. `xi` (evaporation) too high (close to `0.99`) forgets almost
  everything each iteration and behaves close to random search; too low
  and the colony can prematurely converge onto one region.
- **Both:** `random_seed` makes a run fully reproducible — same seed,
  same fitness function, same result every time (this is asserted in
  `run_pso_optimization.py`/`run_aco_optimization.py`'s optional
  `RUN_REPRODUCIBILITY_CHECK`).