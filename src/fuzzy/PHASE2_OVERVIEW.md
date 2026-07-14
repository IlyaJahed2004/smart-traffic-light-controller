# Phase 2 — Fuzzy Controller: Step-by-Step Explanation

**Code:** `src/fuzzy/fuzzy_controller.py`
**Also added:** `src/cost_function.py`
**Test it:** `python experiments/test_fuzzy_controller.py`
**Status:** ✅ Done, bug-fixed and verified — Phase 3 (PSO/ACO) can now build on this

This document walks through **what we built and why**, step by step, in plain language.

---

## Step 1: What problem is this solving?

In Phase 1, we built a traffic simulation where *we* had to manually decide which road gets the green light (we used a dumb "switch every 10 ticks" rule just to test things).

Phase 2's job: replace that dumb rule with a **smart decision-maker**. Given how many cars are waiting on each road right now, it should output: *how long should road 1's green light stay on?*

This decision-maker is a **fuzzy logic controller**.

---

## Step 2: Why "fuzzy" logic instead of simple if/else?

A simple rule might be:
```
if queue_1 > 10: green_1 = 40 seconds
else: green_1 = 10 seconds
```
The problem: what happens right at the boundary — queue_1 = 9 vs queue_1 = 11? The decision suddenly jumps, which is unrealistic and jerky.

Fuzzy logic instead uses **soft categories**: a queue length isn't just "long" or "short" — it can be **70% High and 30% Medium** at the same time. This lets decisions blend smoothly instead of flipping abruptly. That's the whole point of "fuzzy."

---

## Step 3: Designing the inputs and output

Per the project spec, we defined:

- **Inputs (2):** queue length of road 1, queue length of road 2
  - Each is split into 3 categories: `Low`, `Medium`, `High`
- **Output (1):** green time for road 1
  - Split into 3 categories: `Short`, `Medium`, `Long`

**What about road 2's green time?** The spec only defines *one* fuzzy output (road 1's green time). So we derive road 2's green time as: `cycle_time - green_time_1` (with safe min/max limits). This keeps the total green time per cycle roughly constant, which is realistic — think of it as a shared "budget" of green time split between the two roads.

---

## Step 4: How a category like "High" is defined mathematically

Each category (`Low`, `Medium`, `High`, `Short`, `Medium`, `Long`) is defined by a **triangle shape**:

```
membership
   1 |        /\
     |       /  \
     |      /    \
   0 |_____/      \_____
           a   b   c
```

- Below point `a` or above point `c`: membership is 0 (definitely not this category)
- At point `b` (the peak): membership is 1 (definitely this category)
- In between: membership rises or falls in a straight line

Three numbers `(a, b, c)` fully describe one category's shape. This is what makes the whole system **tunable** — change `a`, `b`, `c` and you change what counts as "High."

**Edge case worth knowing:** when `a == b` (e.g. `Low = (0, 0, 7)`) or `b == c` (e.g. `High = (13, 20, 20)`), the shape isn't a full triangle but a "shoulder" — it's already at its peak (1.0) right at the edge of the input range, and only slopes down on one side. This is intentional and common at the boundaries, but it needs careful handling in code (see Step 10).

---

## Step 5: The rule base — the "human knowledge" part

We wrote 9 rules covering every combination of (road 1 queue category, road 2 queue category), following the spec's examples:

| Road 1 queue | Road 2 queue | → Green time for Road 1 |
|---|---|---|
| High | Low | Long |
| Low | High | Short |
| Medium | Medium | Medium |
| ... | ... | ... (9 total, covering all 3×3 combinations) |

This is the "expert knowledge" — written the way a human traffic engineer might reason about it, just formalized.

---

## Step 6: Making the rules "weighted"

The project spec requires that PSO/ACO can tune **rule weights**, not just the category shapes. So each of the 9 rules got a **weight from 0 to 1**:
- Weight = 1 → rule fires at full strength
- Weight = 0 → rule is effectively turned off
- In between → rule contributes partially

This lets the optimization algorithms later discover that some rules matter more than others.

---

## Step 7: How a decision actually gets made (the 4-step pipeline)

Given `queue_1 = 12`, `queue_2 = 3`, here's what happens inside `compute_green_time()`:

1. **Fuzzify** — figure out how much `queue_1=12` belongs to `Low`/`Medium`/`High` (e.g. maybe 20% Medium, 80% High), same for `queue_2=3`.

2. **Evaluate rules** — for each of the 9 rules, compute how strongly it "fires." A rule like "IF road1 High AND road2 Low THEN green1 Long" fires based on the *weaker* of the two matching degrees (this is standard fuzzy AND = minimum), then multiplied by that rule's weight.

3. **Aggregate** — for each output category (`Short`/`Medium`/`Long`), take the strongest rule pointing to it. This builds one combined "fuzzy shape" for the final answer.

4. **Defuzzify (centroid method, as required by the spec)** — convert that fuzzy shape into one crisp number by finding its "center of mass." This is the final `green_time_1` in seconds.

Road 2's time is then `cycle_time - green_time_1`.

This is the full classic Mamdani pipeline — clip each rule's output at its firing strength, take the union of all clipped shapes, then compute the centroid of that combined shape. This is **not** the same as the alternative "Center Average" defuzzification method (weighted average of rule centers, no aggregation step) — the two are alternatives, never combined. See the code's docstring on `compute_green_time()` for the full comparison.

---

## Step 8: Making it tunable for PSO/ACO (the important design step)

Everything tunable is packed into **one flat list of 27 numbers**:

- 9 numbers: shapes of the 3 input categories (`Low`, `Medium`, `High`) × 3 points each
- 9 numbers: shapes of the 3 output categories (`Short`, `Medium`, `Long`) × 3 points each
- 9 numbers: one weight per rule

This single list of 27 numbers is exactly what PSO calls a "particle position" and what ACO calls "a solution an ant builds." We wrote conversion functions so this list can be turned into the readable rule/shape structure and back, without losing any information (verified this round-trip is exact).

We also provide the valid **minimum/maximum bounds** for each of the 27 numbers, so PSO/ACO know what range to search in.

---

## Step 9: The cost function — what PSO/ACO are trying to minimize

Recall from the spec:
```
C = α·W + β·Q + γ·S
```
- W = average waiting time
- Q = average queue length
- S = number of stops

We wrote `cost_function.py` with:
- `compute_cost(metrics, alpha, beta, gamma)` — plugs Phase 1's simulation output directly into this formula
- `evaluate_controller(controller, env_factory, num_steps)` — the **all-in-one function**: takes a fuzzy controller, runs a full traffic simulation with it, and returns its cost. This is the exact function PSO/ACO will call, over and over, once per candidate solution they try.

---

## Step 10: Testing what we built (including a real bug we found and fixed)

We tested (see `experiments/test_fuzzy_controller.py`):
1. **Round-trip check** — converting the 27-number list to rules/shapes and back gives identical numbers (no data lost).
2. **Sample decisions** — gave the controller different queue combinations and printed the actual green-time split, so the decisions are visible, not just a final cost number.
3. **Full pipeline works** — ran the fuzzy controller inside the actual traffic simulation from Phase 1, end-to-end, without errors.
4. **Compared against Phase 1's baseline, across multiple traffic scenarios** — not just one.

### A real bug we found and FIXED (not kept)

While printing sample decisions, we noticed something wrong: an empty road (`queue=0`) paired with a completely full road (`queue=20`) was still getting an even 25/25 green-time split — as if the controller couldn't tell the two roads apart at all.

**Root cause:** the triangular membership function had an edge-case bug. For "shoulder" shapes (see Step 4) where the peak sits exactly on the boundary of the input range (e.g. `High = (13, 20, 20)`), the code checked `x >= c → return 0`, which incorrectly zeroed out the peak itself when `x` landed exactly on that boundary. So `queue=20` was being fuzzified as **0% Low, 0% Medium, 0% High** — belonging to nothing. With no category active, no rules fired, and the code silently fell back to a safe default: the exact midpoint of the output range. That's where the wrong 25/25 split was coming from.

**Fix:** corrected the boundary logic so a shoulder's peak edge correctly returns 1.0. Verified with targeted tests (`fuzzify(20)` and `fuzzify(0)` now correctly return "fully High" / "fully Low").

### Honest evaluation after the fix

After fixing the bug, we compared the fuzzy controller against the fixed-timer baseline across **four traffic scenarios**, not just one:

| Scenario | Fixed-timer cost | Fuzzy cost | Winner |
|---|---|---|---|
| Moderate, symmetric traffic | 63.66 | 84.29 | Fixed-timer |
| Moderate, asymmetric traffic | 71.24 | 61.62 | Fuzzy |
| Heavy, asymmetric traffic | 392.12 | 90.13 | Fuzzy (by a lot) |
| Heavy, symmetric traffic | 192.13 | 190.82 | Fuzzy (barely) |

**This is not a clean win for fuzzy, and we're reporting that honestly.** The fuzzy controller (with default, hand-picked parameters) tends to win under heavy and/or imbalanced traffic — exactly where an adaptive controller should help — but can lose under light, balanced traffic. This is a legitimate, useful baseline result: it shows the default parameters aren't robust across conditions, which is **exactly the problem Phase 3 (PSO/ACO) exists to solve**.

---

## Step 11: What's now possible because of this work

Your teammate (Phase 3) can write PSO and ACO **without understanding any fuzzy logic at all**. All they need is:

```python
controller = FuzzyController()
lower, upper = controller.get_param_bounds()        # search space (27 numbers, min/max each)
controller.set_params_from_vector(candidate)          # try a candidate solution
cost = evaluate_controller(controller, env_factory, num_steps)   # get its score (minimize this)
```

That's the entire interface. Everything about triangles, rules, and centroids is hidden behind it.

**One suggestion for Phase 3:** since the default baseline isn't robust across traffic conditions, consider evaluating each candidate solution across *multiple* traffic scenarios (not just one `env_factory`) so PSO/ACO find parameters that generalize well, rather than overfitting to a single scenario. Worth a quick team discussion before locking in the Phase 3 evaluation strategy.

---

## Summary: Files created in this phase

| File | Purpose |
|---|---|
| `src/fuzzy/fuzzy_controller.py` | The `FuzzyController` class — fuzzify, evaluate rules, defuzzify, plus the vector interface for PSO/ACO |
| `src/cost_function.py` | Turns simulation metrics into the single cost number `C = α·W + β·Q + γ·S` |
| `experiments/test_fuzzy_controller.py` | Smoke test: sample decisions, full pipeline check, and multi-scenario comparison against the Phase 1 baseline |
| `src/fuzzy/README.md` | Technical reference doc for Phase 3, focused on the handoff interface and the honest evaluation table |
| `src/fuzzy/PHASE2_OVERVIEW.md` | This document — the step-by-step narrative of how and why Phase 2 was built |