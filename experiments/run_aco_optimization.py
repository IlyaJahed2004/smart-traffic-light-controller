"""
run_aco_optimization.py

Final Phase 3 run script for the discretized Ant System ACO
(see optimization/aco.py's module docstring for why it's discretized,
and how archive_size/q/xi were re-purposed from the old ACO_R version).

Mirrors run_pso_optimization.py in setup (same scenarios, same seeds,
same fitness function, same evaluation budget per QUICK_TEST/full
mode) so PSO and ACO results are directly comparable in Phase 4 --
neither algorithm gets an easier or different evaluation.

IMPORTANT -- updated for the new aco.py internals:
This version no longer calls `aco._initialize_archive()` or reads
`aco.archive_positions` / `aco.archive_fitness` -- those belonged to
the old ACO_R implementation and no longer exist. The new
ACOOptimizer keeps its own running best as `aco.best_position` /
`aco.best_cost`, updated via `_update_global_best()`, and its
"pheromone update" (evaporation + deposit) via `_update_pheromone()`
-- called in that order, exactly matching the class's own
`optimize()` method internals. This script reproduces that same
sequence manually purely to print progress + ETA after every
iteration, since `optimize()` itself is a black box that only
returns once at the very end.

--------------------------------------------------------------------
ACO AND THE BASELINE (seed_with_default_vector=True, but SOFT)
--------------------------------------------------------------------
Both ACO_KWARGS below set seed_with_default_vector=True, same as
run_pso_optimization.py. Unlike PSO, this does NOT force the baseline
vector into the ant population -- it only pre-boosts pheromone near
the baseline's nearest bin in every dimension by a fixed factor
(_bias_toward_default_vector, x5). Ant selection stays fully
probabilistic (roulette-wheel over tau ** alpha), so:

    ACO's best_cost is USUALLY <= baseline_cost with this bias on and
    enough ants/iterations, but this is NOT a hard guarantee the way
    it is for PSO.

Two consequences, both handled below:
  1. QUICK_TEST's budget was bumped (see below) specifically so this
     soft bias has enough ants/iterations to actually pay off, rather
     than a handful of unlucky draws never finding the primed region.
  2. main() prints a WARNING (not an assertion) if ACO finishes worse
     than baseline, since -- unlike PSO -- that outcome is possible
     without indicating a bug, just bad luck or too small a budget.

--------------------------------------------------------------------
QUICK_TEST MODE (default: on)
--------------------------------------------------------------------
This script defaults to a small, fast smoke-test configuration --
1 scenario, 1 seed, 200-step simulations, and a small archive/ant/
iteration count -- so you can sanity-check that ACO runs end-to-end,
respects bounds, produces valid triangles, and (usually) improves on
baseline, in well under a minute instead of ~10 minutes. The budget
here (12-ant priming batch, 10 ants/iteration x 15 iterations = 162
evaluations) matches run_pso_optimization.py's QUICK_TEST evaluation
count so the two scripts' smoke tests are directly comparable, not
just individually "fast."

Set QUICK_TEST = False to restore the full Phase 4 configuration
(4 scenarios x 3 seeds, 1000-step simulations, archive_size=30,
num_ants=15, max_iter=100) before you run the real tuning pass whose
results you intend to keep / compare against PSO.
--------------------------------------------------------------------

Run from the project root:
    python experiments/run_aco_optimization.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import evaluate_controller
from optimization.aco import ACOOptimizer, make_multi_scenario_fitness

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

# --- toggle between the quick smoke test and the full Phase 4 run ---
QUICK_TEST = True

# Set to True to also re-run the whole optimization a second time with
# the same seed, to verify reproducibility. Doubles total runtime --
# off by default, mirroring run_pso_optimization.py's flag, so a
# straight PSO-vs-ACO timing comparison in Phase 4 isn't skewed by one
# script silently doing twice the work of the other.
RUN_REPRODUCIBILITY_CHECK = False

if QUICK_TEST:
    # Small enough to finish in well under a minute, but bumped up from
    # an earlier bare-minimum smoke-test budget so the soft baseline
    # bias (see module docstring) has a realistic chance to pay off,
    # not just prove the pipeline runs. Evaluation count (162) is kept
    # close to PSO's QUICK_TEST budget (180) for a fair side-by-side
    # smoke test.
    SCENARIOS = [
        ("moderate symmetric", 0.3, 0.3),
    ]
    SEEDS = [1]
    NUM_STEPS = 200

    ACO_KWARGS = dict(
        archive_size=12,
        num_ants=10,
        max_iter=15,
        q=0.3,
        xi=0.85,
        num_bins=20,
        alpha=1.0,
        tau0=1.0,
        tau_min=1e-3,
        elitist_weight=2.0,
        seed_with_default_vector=True,
        random_seed=4,
    )
else:
    # Full Phase 4 configuration -- identical to run_pso_optimization.py
    # on purpose, so PSO and ACO get the exact same evaluation.
    SCENARIOS = [
        ("moderate symmetric",  0.3, 0.3),
        ("moderate asymmetric", 0.4, 0.2),
        ("heavy asymmetric",    0.6, 0.2),
        ("heavy symmetric",     0.5, 0.5),
    ]
    SEEDS = [1, 2, 3]
    NUM_STEPS = 1000

    ACO_KWARGS = dict(
        archive_size=30,   # size of the initial random priming batch (not a
                            # persistent archive anymore -- see aco.py docstring)
        num_ants=15,        # ants constructed per iteration
        max_iter=100,
        q=0.3,              # re-purposed: Ant-System deposit constant Q
                            # (Delta_tau = Q / cost), NOT the old ACO_R
                            # locality parameter
        xi=0.85,            # re-purposed: pheromone evaporation rate rho
                            # in [0.01, 0.99], NOT the old ACO_R spread decay
        num_bins=20,        # discretization resolution per dimension
        alpha=1.0,          # pheromone-influence exponent (selection prob ~ tau**alpha)
        tau0=1.0,           # initial pheromone value in every cell
        tau_min=1e-3,       # pheromone floor -- keeps exploration alive (Max-Min style)
        elitist_weight=2.0, # extra deposit multiplier for the global-best solution
        # Unlike PSO (where this GUARANTEES one particle starts exactly at
        # the default vector), this only pre-boosts the pheromone near the
        # default vector's nearest bins -- a soft bias, not a guarantee the
        # default vector is ever actually sampled. Default for this class
        # is False; turned on here for rough parity with PSO's behavior.
        seed_with_default_vector=True,
        random_seed=7,
    )


def make_env_factory(r1, r2, seed):
    """Returns a zero-arg callable that builds a fresh TrafficEnv.
    Defaults on r1/r2/seed avoid the classic late-binding closure bug
    when this is called inside a loop."""
    def _factory(r1=r1, r2=r2, seed=seed):
        return TrafficEnv(arrival_rate_1=r1, arrival_rate_2=r2,
                           departure_rate=1.0, seed=seed)
    return _factory


def build_env_factories():
    """Cross product of SCENARIOS x SEEDS -> list of env_factories."""
    factories = []
    labels = []
    for label, r1, r2 in SCENARIOS:
        for seed in SEEDS:
            factories.append(make_env_factory(r1, r2, seed))
            labels.append(f"{label} (seed={seed})")
    return factories, labels


def check_feasibility(position, lower, upper, label):
    assert ((position >= lower - 1e-9) & (position <= upper + 1e-9)).all(), \
        f"[{label}] position out of bounds"
    for start in range(0, 18, 3):
        a, b, c = position[start:start + 3]
        assert a <= b + 1e-9 <= c + 1e-9, f"[{label}] invalid triangle at {start}: {a},{b},{c}"
    print(f"[{label}] OK: within bounds, all triangles valid")


def vector_to_readable(controller: FuzzyController, vector: np.ndarray) -> dict:
    """Same grouping compare_algorithms.py uses: MF breakpoints and rule
    weights keyed by human-readable names instead of a bare 27-vector,
    so the logged JSON is inspectable without decoding indices by hand."""
    params = controller.vector_to_params(vector)
    readable = {
        "mf_queue": {k: [round(float(x), 4) for x in v] for k, v in params["mf_queue"].items()},
        "mf_green": {k: [round(float(x), 4) for x in v] for k, v in params["mf_green"].items()},
        "rule_weights": {},
    }
    for i, (s1, s2, out) in enumerate(FuzzyController.RULES):
        label = f"R{i}: IF q1={s1} AND q2={s2} THEN green={out}"
        readable["rule_weights"][label] = round(float(params["rule_weights"][i]), 4)
    return readable


def save_final_vector(controller, best_position, best_cost, baseline_cost, history,
                       out_dir: Path = DEFAULT_LOG_DIR, timestamp: str | None = None) -> tuple[Path, Path]:
    """
    Persist the tuned vector to disk in two forms:
      - a raw .npy array, for loading straight back into a controller
        via controller.set_params_from_vector(np.load(path)), no
        parsing needed.
      - a readable .json with the vector grouped by MF/rule name, plus
        the run's config and resulting cost, so a saved file is
        self-describing months later.

    Filenames are tagged quick/full (from QUICK_TEST) and timestamped,
    so a quick smoke-test run can never silently overwrite a real
    tuning run's results. Pass the same `timestamp` used for
    save_convergence_plot() so the two output files from one run share
    a matching stem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if QUICK_TEST else "full"
    stem = f"aco_best_vector_{mode}_{timestamp}"

    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, best_position)

    improvement = baseline_cost - best_cost
    payload = {
        "timestamp": timestamp,
        "mode": mode,
        "quick_test": QUICK_TEST,
        "best_cost": best_cost,
        "baseline_cost": baseline_cost,
        "improvement": improvement,
        "improvement_pct": (100 * improvement / baseline_cost) if baseline_cost else None,
        "num_iterations_run": len(history) - 1,
        "aco_kwargs": ACO_KWARGS,
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "num_steps": NUM_STEPS,
        "best_position_raw": [float(x) for x in best_position],
        "best_position_readable": vector_to_readable(controller, best_position),
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved final vector to:\n  {npy_path}\n  {json_path}")
    return npy_path, json_path


def save_convergence_plot(history, baseline_cost, out_dir: Path = DEFAULT_PLOTS_DIR,
                           timestamp: str | None = None) -> Path:
    """
    Plot best-cost-so-far per iteration against the baseline -- same
    style as compare_optimizers.py's plot_convergence(), single-
    algorithm version (that script overlays both PSO and ACO on one
    figure; this one just has ACO, since run_pso_optimization.py saves
    its own equivalent plot separately).

    Saved under results/plots/, tagged quick/full and timestamped like
    save_final_vector()'s files, so a smoke-test plot never overwrites
    a real run's plot. Pass the same `timestamp` used for
    save_final_vector() so the two files from one run share a
    matching stem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if QUICK_TEST else "full"
    out_path = out_dir / f"aco_convergence_{mode}_{timestamp}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(history, label=f"ACO (best={history[-1]:.3f})")
    plt.axhline(baseline_cost, color="gray", linestyle="--",
                label=f"Baseline ({baseline_cost:.3f})")
    plt.xlabel("Iteration")
    plt.ylabel("Best avg cost so far")
    plt.title("ACO convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved convergence plot to {out_path}")
    return out_path


def report_per_scenario(controller, factories, labels):
    """Print the tuned controller's cost on each individual scenario,
    not just the aggregate -- so a low average can't hide one scenario
    that got much worse."""
    print("\nPer-scenario breakdown (tuned controller):")
    for factory, label in zip(factories, labels):
        cost = evaluate_controller(controller, factory, NUM_STEPS)
        print(f"  {label:30s} cost={cost:8.2f}")


def main():
    if QUICK_TEST:
        print("QUICK_TEST mode is ON -- this is a fast pipeline smoke test, "
              "not a real tuning run. Set QUICK_TEST = False for the full "
              "Phase 4 configuration.\n")

    # Shared across save_convergence_plot() and save_final_vector() so
    # this run's plot and logged vector share a matching filename stem.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    env_factories, labels = build_env_factories()
    print(f"Optimizing against {len(env_factories)} scenario/seed combinations "
          f"({len(SCENARIOS)} traffic patterns x {len(SEEDS)} seeds each)\n")

    controller = FuzzyController()
    lower, upper = controller.get_param_bounds()

    fitness_fn = make_multi_scenario_fitness(
        env_factories=env_factories,
        num_steps=NUM_STEPS,
        aggregate=np.mean,
    )

    # --- baseline: default hand-picked params, same robust fitness ---
    default_vec = controller.get_default_vector()
    controller.set_params_from_vector(default_vec)
    baseline_cost = fitness_fn(controller)
    print(f"Baseline (default vector) avg cost across all scenarios: {baseline_cost:.4f}")

    # --- ACO run (manual loop instead of aco.optimize(), so we can
    #     print progress + ETA after every iteration -- optimize()
    #     itself is a black box that only returns once at the end).
    #     This reproduces optimize()'s exact sequence: construct ants,
    #     update global best, THEN update pheromone (evaporate+deposit),
    #     in that order, both for the initial priming batch and every
    #     subsequent iteration. ---
    aco = ACOOptimizer(controller, fitness_fn, **ACO_KWARGS)

    sims_per_ant = len(env_factories)
    init_sims = ACO_KWARGS["archive_size"] * sims_per_ant
    iter_sims = ACO_KWARGS["num_ants"] * sims_per_ant

    print(f"\nPriming pheromone table ({ACO_KWARGS['archive_size']} random ants, "
          f"{len(env_factories)} scenarios each = {init_sims} simulations)...")
    if ACO_KWARGS.get("seed_with_default_vector"):
        print("  seed_with_default_vector=True: pheromone pre-boosted near the "
              "baseline vector's bins (a soft bias, not a guarantee -- ACO "
              "selection stays probabilistic, unlike PSO's hard seeding).")
    t_start = time.time()
    init_ants = [aco._construct_ant() for _ in range(aco.init_batch)]
    aco._update_global_best(init_ants)
    aco._update_pheromone(init_ants)
    t_init = time.time() - t_start
    history = [aco.best_cost]
    print(f"  done in {t_init:.1f}s. Initial best avg cost: {aco.best_cost:.4f}")

    print(f"\nRunning {ACO_KWARGS['max_iter']} iterations "
          f"({ACO_KWARGS['num_ants']} ants x {len(env_factories)} scenarios = "
          f"{iter_sims} simulations/iteration)...")
    for it in range(ACO_KWARGS["max_iter"]):
        t_iter_start = time.time()
        ants = [aco._construct_ant() for _ in range(aco.num_ants)]
        aco._update_global_best(ants)
        aco._update_pheromone(ants)
        history.append(aco.best_cost)

        t_iter = time.time() - t_iter_start
        elapsed = time.time() - t_start
        remaining_iters = ACO_KWARGS["max_iter"] - (it + 1)
        eta = remaining_iters * t_iter
        print(f"  iter {it + 1:>3}/{ACO_KWARGS['max_iter']}  "
              f"best={aco.best_cost:9.4f}  "
              f"({t_iter:5.1f}s this iter, {elapsed / 60:5.1f}min elapsed, "
              f"~{eta / 60:5.1f}min remaining)")

    best_position = aco.best_position.copy()
    best_cost = aco.best_cost

    print(f"\nACO best avg cost:                {best_cost:.4f}")
    print(f"Improvement over baseline:        {baseline_cost - best_cost:+.4f} "
          f"({100 * (baseline_cost - best_cost) / baseline_cost:.1f}%)")
    print(f"Convergence (every 10th entry):   {[round(h, 3) for h in history[::10]]}")

    # best_cost is tracked separately from the pheromone table (which
    # keeps evaporating/being reinforced), and only ever updated on
    # strict improvement in _update_global_best -- so it can never
    # regress, the same guarantee PSO's global best has.
    monotonic = all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1))
    assert monotonic, "history is not monotonically non-increasing (global best regressed)"
    print("Monotonic convergence OK (global best never regresses)")

    save_convergence_plot(history, baseline_cost, timestamp=run_timestamp)

    # SOFT baseline check: unlike PSO, seed_with_default_vector=True
    # for ACO only biases pheromone near the baseline -- it does NOT
    # force the baseline vector into the ant population. So ending up
    # worse than baseline is POSSIBLE here (unlucky draws, too small a
    # budget) without indicating a bug the way it would for PSO. Warn,
    # don't assert.
    if ACO_KWARGS.get("seed_with_default_vector"):
        if best_cost <= baseline_cost + 1e-9:
            print("Baseline comparison OK: best_cost <= baseline_cost.")
        else:
            print(f"WARNING: ACO finished worse than baseline "
                  f"(best={best_cost:.4f} > baseline={baseline_cost:.4f}). "
                  f"This is possible (not a bug) since ACO's baseline bias is "
                  f"soft, not a hard guarantee like PSO's -- but if this "
                  f"happens consistently, try increasing num_ants/max_iter/"
                  f"archive_size, or raising the bias boost_factor in "
                  f"_bias_toward_default_vector (aco.py).")

    check_feasibility(best_position, lower, upper, "best_position")
    assert best_position.shape == (27,), "vector length must be 27"

    # --- reproducibility check: same seed -> identical result (optional,
    #     off by default since it doubles total runtime -- see flag above).
    #     Uses the class's own optimize() end-to-end, since that's still
    #     the public black-box contract and is unaffected by the manual
    #     -loop reproduction above. ---
    if RUN_REPRODUCIBILITY_CHECK:
        print("\nRe-running once more with the same seed to verify reproducibility "
              "(this doubles total runtime)...")
        controller_repeat = FuzzyController()
        aco_repeat = ACOOptimizer(controller_repeat, fitness_fn, **ACO_KWARGS)
        best_position_2, best_cost_2, _ = aco_repeat.optimize()
        assert best_cost == best_cost_2, "same seed produced different cost"
        assert np.allclose(best_position, best_position_2), "same seed produced different position"
        print("Reproducibility OK: identical seed -> identical result")
    else:
        print("\n(Skipping reproducibility re-run -- set RUN_REPRODUCIBILITY_CHECK = True "
              "to enable it.)")

    # --- make sure the win isn't hiding a regression on one scenario ---
    controller.set_params_from_vector(best_position)
    report_per_scenario(controller, env_factories, labels)

    save_final_vector(controller, best_position, best_cost, baseline_cost, history,
                       timestamp=run_timestamp)

    print("\nAll checks passed. `best_position` is ready to be loaded as the")
    print("final tuned controller (controller.set_params_from_vector(best_position)).")


if __name__ == "__main__":
    main()