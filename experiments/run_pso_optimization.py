"""
run_pso_optimization.py

Final Phase 3 run script for PSO. Unlike the earlier smoke-test
version (which used a single fixed-seed scenario just to check the
pipeline didn't crash), this version optimizes against a ROBUST
fitness: the average cost across several traffic scenarios (light/
heavy, symmetric/asymmetric), each repeated over several random
seeds, so the tuned controller isn't just overfit to one specific
arrival sequence.

This is computationally heavy (30 particles x 12 scenario/seed
combos x 100 iterations = ~36,000 full 1000-tick simulations), so
this version prints progress after every iteration with a running
ETA -- it is NOT frozen if you don't see new output for a while,
each individual iteration just takes a while.

--------------------------------------------------------------------
QUICK_TEST MODE (default: on)
--------------------------------------------------------------------
This script defaults to a small, fast smoke-test configuration --
1 scenario, 1 seed, 200-step simulations, and a small swarm/iteration
count -- so you can sanity-check that PSO runs end-to-end, respects
bounds, produces valid triangles, and improves on baseline, in well
under a minute instead of the full multi-hour run.

Set QUICK_TEST = False to restore the full Phase 4 configuration
(4 scenarios x 3 seeds, 1000-step simulations, num_particles=30,
max_iter=100) before you run the real tuning pass whose results you
intend to keep / compare against ACO.
--------------------------------------------------------------------

Run from the project root:
    python experiments/run_pso_optimization.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import evaluate_controller
from optimization.pso import PSOOptimizer, make_multi_scenario_fitness

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "results" / "logs"

# --- toggle between the quick smoke test and the full Phase 4 run ---
QUICK_TEST = True

# Set to True to also re-run the whole optimization a second time with
# the same seed, to verify reproducibility. Doubles total runtime --
# off by default since it's a one-time sanity check, not something you
# need on every run once you've confirmed it once.
RUN_REPRODUCIBILITY_CHECK = False

if QUICK_TEST:
    # Small enough to finish in well under a minute. Not meant to
    # produce a genuinely well-tuned controller -- just to verify the
    # pipeline (bounds, triangle validity, monotonic convergence,
    # reproducibility hook) works end-to-end before committing to the
    # full run below.
    SCENARIOS = [
        ("moderate symmetric", 0.3, 0.3),
    ]
    SEEDS = [1]
    NUM_STEPS = 200

    PSO_KWARGS = dict(
        num_particles=8, max_iter=10,
        w=0.7, w_min=0.4, c1=1.5, c2=1.5,
        random_seed=7,
    )
else:
    SCENARIOS = [
        ("moderate symmetric",  0.3, 0.3),
        ("moderate asymmetric", 0.4, 0.2),
        ("heavy asymmetric",    0.6, 0.2),
        ("heavy symmetric",     0.5, 0.5),
    ]
    SEEDS = [1, 2, 3]
    NUM_STEPS = 1000

    PSO_KWARGS = dict(
        num_particles=30, max_iter=100,
        w=0.7, w_min=0.4, c1=1.5, c2=1.5,
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
                       out_dir: Path = DEFAULT_LOG_DIR) -> tuple[Path, Path]:
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
    tuning run's results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "quick" if QUICK_TEST else "full"
    stem = f"pso_best_vector_{mode}_{timestamp}"

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
        "pso_kwargs": PSO_KWARGS,
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

    # --- PSO run (manual loop instead of pso.optimize(), so we can
    #     print progress + ETA after every iteration -- optimize()
    #     itself is a black box that only returns once at the end) ---
    pso = PSOOptimizer(controller, fitness_fn, **PSO_KWARGS)

    print(f"\nInitializing swarm ({PSO_KWARGS['num_particles']} particles, "
          f"{len(env_factories)} scenarios each = "
          f"{PSO_KWARGS['num_particles'] * len(env_factories)} simulations)...")
    t_start = time.time()
    pso._initialize_population()
    t_init = time.time() - t_start
    history = [pso.global_best_cost]
    print(f"  done in {t_init:.1f}s. Initial best avg cost: {pso.global_best_cost:.4f}")

    print(f"\nRunning {PSO_KWARGS['max_iter']} iterations "
          f"({PSO_KWARGS['num_particles'] * len(env_factories)} simulations/iteration)...")
    for it in range(PSO_KWARGS["max_iter"]):
        t_iter_start = time.time()
        inertia = pso._current_inertia(it)
        for particle in pso.swarm:
            pso._update_velocity(particle, inertia)
            pso._update_position(particle)
            particle.fitness = pso._evaluate(particle.position)
            pso._update_personal_best(particle)
            pso._update_global_best(particle)
        history.append(pso.global_best_cost)

        t_iter = time.time() - t_iter_start
        elapsed = time.time() - t_start
        remaining_iters = PSO_KWARGS["max_iter"] - (it + 1)
        eta = remaining_iters * t_iter
        print(f"  iter {it + 1:>3}/{PSO_KWARGS['max_iter']}  "
              f"best={pso.global_best_cost:9.4f}  "
              f"({t_iter:5.1f}s this iter, {elapsed / 60:5.1f}min elapsed, "
              f"~{eta / 60:5.1f}min remaining)")

    best_position = pso.global_best_position
    best_cost = pso.global_best_cost

    print(f"\nPSO best avg cost:                {best_cost:.4f}")
    print(f"Improvement over baseline:        {baseline_cost - best_cost:+.4f} "
          f"({100 * (baseline_cost - best_cost) / baseline_cost:.1f}%)")
    print(f"Convergence (every 10th entry):   {[round(h, 3) for h in history[::10]]}")

    # PSO's global best only ever updates on strict improvement, so the
    # recorded history can never get worse from one iteration to the next.
    monotonic = all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1))
    assert monotonic, "history is not monotonically non-increasing (global best regressed)"
    print("Monotonic convergence OK (global best never regresses)")

    check_feasibility(best_position, lower, upper, "best_position")
    assert best_position.shape == (27,), "vector length must be 27"

    # --- reproducibility check: same seed -> identical result (optional,
    #     off by default since it doubles total runtime -- see flag above) ---
    if RUN_REPRODUCIBILITY_CHECK:
        print("\nRe-running once more with the same seed to verify reproducibility "
              "(this doubles total runtime)...")
        controller_repeat = FuzzyController()
        pso_repeat = PSOOptimizer(controller_repeat, fitness_fn, **PSO_KWARGS)
        best_position_2, best_cost_2, _ = pso_repeat.optimize()
        assert best_cost == best_cost_2, "same seed produced different cost"
        assert np.allclose(best_position, best_position_2), "same seed produced different position"
        print("Reproducibility OK: identical seed -> identical result")
    else:
        print("\n(Skipping reproducibility re-run -- set RUN_REPRODUCIBILITY_CHECK = True "
              "to enable it.)")

    # --- make sure the win isn't hiding a regression on one scenario ---
    controller.set_params_from_vector(best_position)
    report_per_scenario(controller, env_factories, labels)

    save_final_vector(controller, best_position, best_cost, baseline_cost, history)

    print("\nAll checks passed. `best_position` is ready to be loaded as the")
    print("final tuned controller (controller.set_params_from_vector(best_position)).")


if __name__ == "__main__":
    main()