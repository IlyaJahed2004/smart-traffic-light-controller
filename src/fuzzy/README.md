# Phase 2 — Fuzzy Controller

**Code:** `src/fuzzy/fuzzy_controller.py`
**Also added:** `src/cost_function.py`
**Test it:** `python experiments/test_fuzzy_controller.py`
**Status:** ✅ Core implementation done, bug-fixed and verified — ready for Phase 3 (PSO/ACO) to build on

## What this is, in plain words

This is the "brain" that looks at how many cars are waiting on each road and decides how long the green light should stay on. It uses **fuzzy logic**: instead of a hard rule like "if queue > 10, switch," it uses smooth categories — `Low`, `Medium`, `High` — so decisions don't awkwardly flip-flop right at a boundary.

**Inputs:** queue length of road 1, queue length of road 2
**Output:** green time for road 1 (in seconds). Road 2's green time is just `cycle_time - green_time_1` (clipped to valid bounds), since the project spec only defines one fuzzy output.

## ⚠️ Honest evaluation: the baseline is NOT universally better than a fixed timer

We tested the default (hand-picked) fuzzy controller against a naive fixed-timer baseline across several traffic scenarios. Results:

| Scenario | Arrival rates (r1, r2) | Fixed-timer cost | Fuzzy cost | Winner |
|---|---|---|---|---|
| Moderate, symmetric | (0.3, 0.3) | 63.66 | 84.29 | **Fixed-timer** |
| Moderate, asymmetric | (0.4, 0.2) | 71.24 | 61.62 | **Fuzzy** |
| Heavy, asymmetric | (0.6, 0.2) | 392.12 | 90.13 | **Fuzzy** (by a lot) |
| Heavy, symmetric | (0.5, 0.5) | 192.13 | 190.82 | **Fuzzy** (barely) |

**Takeaway:** the fuzzy controller (with default, hand-picked parameters) tends to win under **heavy and/or imbalanced traffic** — exactly where an adaptive controller should help, since a fixed-timer wastes green time on an empty/light road. But under **light, balanced traffic**, it can actually be worse than simple alternation.

This is not a bug — it's a legitimate, informative baseline result. It shows the default parameters aren't robust across conditions, which is **exactly what Phase 3 (PSO/ACO) is supposed to fix**: find parameters that perform well across a range of traffic scenarios, not just one.

You can reproduce this table any time by running `python experiments/test_fuzzy_controller.py` — the multi-scenario comparison runs automatically at the end.

## 📌 If you're doing Phase 3 (PSO / ACO), read this section only

You don't need to understand any fuzzy logic internals. Here's the entire contract:

```python
from fuzzy.fuzzy_controller import FuzzyController

controller = FuzzyController()

# 1. Get the search space bounds (one min/max pair per tunable number)
lower, upper = controller.get_param_bounds()   # numpy arrays, same length

# 2. Get a starting point (the hand-designed baseline)
default_vector = controller.get_default_vector()   # numpy array, 27 numbers

# 3. Your algorithm generates candidate vectors (particle positions /
#    ant-constructed solutions) within [lower, upper]. For each candidate:
controller.set_params_from_vector(candidate_vector)

# 4. Evaluate that candidate by running a full simulation and getting its cost:
from cost_function import evaluate_controller
from simulation.traffic_env import TrafficEnv

env_factory = lambda: TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2,
                                   departure_rate=1.0, seed=42)
cost = evaluate_controller(controller, env_factory, num_steps=2000, decision_interval=1)

# 5. Your algorithm wants to MINIMIZE `cost`. That's it.
```

That's the whole interface. `cost` is `C = α·W + β·Q + γ·S` computed automatically by `cost_function.py` — you don't need to compute it by hand.

**Recommendation:** since the default baseline is not robust across traffic conditions (see table above), consider evaluating candidate solutions across *multiple* `env_factory` scenarios (e.g. average their cost) rather than a single fixed one, so PSO/ACO find parameters that generalize rather than overfitting to one specific traffic pattern. This is worth discussing with your teammate before locking in the Phase 3 evaluation strategy.

### What's inside the vector (27 numbers)

You don't need to know this to write PSO/ACO — `get_param_bounds()` and `set_params_from_vector()` handle it — but for reference:
- 9 numbers: shapes of the 3 input membership functions (`Low`, `Medium`, `High` for queue length) — 3 points each
- 9 numbers: shapes of the 3 output membership functions (`Short`, `Medium`, `Long` for green time) — 3 points each
- 9 numbers: one weight (0 to 1) per fuzzy rule, controlling how strongly that rule contributes

This matches the spec exactly: *"position of each particle contains membership function parameters and fuzzy rule weights."*

### Important: `decision_interval`

`evaluate_controller(..., decision_interval=N)` controls how often (in ticks) the controller is allowed to re-decide which road is green.
- `decision_interval=1` — re-decide every tick. This is what all testing above uses, and is confirmed to work correctly.
- Larger values (e.g. 10) are more physically realistic (real lights don't flip every second) but require well-tuned membership functions to avoid starving one road — worth testing carefully if you use this.

## How it works internally (optional reading)

1. **Fuzzify** — each queue length gets a "degree of membership" (0 to 1) in `Low`, `Medium`, `High`, based on triangular membership functions.
2. **Rule evaluation** — 9 rules cover every combination of (queue1, queue2), e.g. "if queue1 is High and queue2 is Low, green1 should be Long." Each rule's strength = `min(degree1, degree2) × rule_weight`.
3. **Aggregation** — for each output category (`Short`/`Medium`/`Long`), take the strongest rule that points to it (union via max).
4. **Defuzzification (centroid)** — the aggregated fuzzy output shape is converted into one crisp number (seconds) by computing its center of mass.

This is the full classic Mamdani pipeline (clip → aggregate → centroid), matching the course slides exactly — **not** combined with the alternative "Center Average" method also shown in the slides. See the docstring on `compute_green_time()` in the code for a detailed note on this distinction.

## Key methods

| Method | What it does, simply |
|---|---|
| `compute_green_time(q1, q2)` | The main decision function: give it two queue lengths, get back `(green_1, green_2)` in seconds. |
| `get_default_params()` | The hand-designed starting parameters, as a readable nested dict. |
| `get_default_vector()` | Same thing, flattened into the 27-number array PSO/ACO use. |
| `get_param_bounds()` | The valid min/max range for each of the 27 numbers. |
| `set_params_from_vector(vec)` | Load a candidate solution into the controller before evaluating it. |
| `params_to_vector(params)` / `vector_to_params(vec)` | Convert between the readable dict form and the flat array form. |

## `cost_function.py`

- `compute_cost(metrics, alpha, beta, gamma)` — plugs `TrafficEnv.get_metrics()` output into `C = α·W + β·Q + γ·S`.
- `evaluate_controller(controller, env_factory, num_steps, ...)` — the all-in-one function: runs a full simulation with a given controller and returns its cost. **This is the function PSO/ACO will call once per candidate.**
- Default weights: `alpha=1.0, beta=1.0, gamma=0.1` (gamma is smaller because `num_stops` tends to be a much bigger raw number than W or Q — adjust these together during Phase 4 experiments).

## Bug fixed during testing

We found and fixed a boundary bug in the triangular membership function: extreme queue values (exactly 0 or exactly the max) were incorrectly fuzzified as belonging to *no* category, causing no rules to fire and the controller to silently default to an equal 25/25 green-time split — even when one road was completely empty. This has been fixed (see git history / commit messages for `fuzzy_controller.py`) and verified with targeted tests.

## Testing done

- Parameter vector round-trips exactly (`vector → params → vector` gives identical numbers).
- `compute_green_time` behaves sensibly at both normal and boundary queue values, giving more green time to whichever road has the longer queue.
- Full pipeline (`FuzzyController` → `TrafficEnv` → `cost_function`) runs end-to-end without errors.
- **Multi-scenario comparison against the Phase 1 fixed-timer baseline** — see table above. Results are mixed and honestly reported, not cherry-picked.

## What's NOT done yet / open for discussion

- Rule *weights* currently only scale how strongly a rule fires — they don't change the rule's structure or consequent.
- No visualization of membership functions yet (useful for the report — could add a plotting script in Phase 4).
- Whether Phase 3 should evaluate candidates on a single traffic scenario or averaged across several (see recommendation above) — worth a quick team discussion before Phase 3 evaluation code is finalized.