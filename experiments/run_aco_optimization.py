"""
run_aco_optimization.py

Final Phase 3 run script for ACO_R. Mirrors run_pso_optimization.py
exactly in setup (same scenarios, same seeds, same fitness function)
so PSO and ACO results are directly comparable in Phase 4 -- neither
algorithm gets an easier or different evaluation.

Like run_pso_optimization.py, this drives ACOOptimizer with a manual
loop instead of calling aco.optimize() directly, purely so we can
print progress + a running ETA after every iteration -- optimize()
itself is a black box that only returns once at the very end, which
is fine for a smoke test but not for a run heavy enough to take
minutes (30 archive members/particles-equivalent x 12 scenario/seed
combos x 100 iterations = thousands of full 1000-tick simulations).
It is NOT frozen if you don't see new output for a while, each
individual iteration just takes a while. If you want to sanity-check
the script quickly before committing to a full run, shrink SEEDS to
a single seed and/or ACO_KWARGS's archive_size/num_ants/max_iter
temporarily.

Run from the project root:
    python experiments/run_aco_optimization.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import evaluate_controller
from optimization.aco import ACOOptimizer, make_multi_scenario_fitness

# Set to True to also re-run the whole optimization a second time with
# the same seed, to verify reproducibility. Doubles total runtime --
# off by default, mirroring run_pso_optimization.py's flag, so a
# straight PSO-vs-ACO timing comparison in Phase 4 isn't skewed by one
# script silently doing twice the work of the other.
RUN_REPRODUCIBILITY_CHECK = False

ACO_KWARGS = dict(
    archive_size=30, num_ants=15, max_iter=100,
    q=0.3, xi=0.85,
    random_seed=7,
)


# ----------------------------------------------------------------------
# Scenario definitions -- identical to run_pso_optimization.py on purpose
# ----------------------------------------------------------------------

SCENARIOS = [
    ("moderate symmetric",  0.3, 0.3),
    ("moderate asymmetric", 0.4, 0.2),
    ("heavy asymmetric",    0.6, 0.2),
    ("heavy symmetric",     0.5, 0.5),
]
SEEDS = [1, 2, 3]
NUM_STEPS = 1000


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


def report_per_scenario(controller, factories, labels):
    """Print the tuned controller's cost on each individual scenario,
    not just the aggregate -- so a low average can't hide one scenario
    that got much worse."""
    print("\nPer-scenario breakdown (tuned controller):")
    for factory, label in zip(factories, labels):
        cost = evaluate_controller(controller, factory, NUM_STEPS)
        print(f"  {label:30s} cost={cost:8.2f}")


def main():
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
    #     itself is a black box that only returns once at the end) ---
    aco = ACOOptimizer(controller, fitness_fn, **ACO_KWARGS)

    sims_per_archive_member = len(env_factories)
    init_sims = ACO_KWARGS["archive_size"] * sims_per_archive_member
    iter_sims = ACO_KWARGS["num_ants"] * sims_per_archive_member

    print(f"\nInitializing archive ({ACO_KWARGS['archive_size']} members, "
          f"{len(env_factories)} scenarios each = {init_sims} simulations)...")
    t_start = time.time()
    aco._initialize_archive()
    t_init = time.time() - t_start
    history = [float(aco.archive_fitness[0])]
    print(f"  done in {t_init:.1f}s. Initial best avg cost: {aco.archive_fitness[0]:.4f}")

    print(f"\nRunning {ACO_KWARGS['max_iter']} iterations "
          f"({ACO_KWARGS['num_ants']} ants x {len(env_factories)} scenarios = "
          f"{iter_sims} simulations/iteration)...")
    for it in range(ACO_KWARGS["max_iter"]):
        t_iter_start = time.time()
        new_ants = [aco._construct_ant() for _ in range(aco.num_ants)]
        aco._update_archive(new_ants)
        history.append(float(aco.archive_fitness[0]))

        t_iter = time.time() - t_iter_start
        elapsed = time.time() - t_start
        remaining_iters = ACO_KWARGS["max_iter"] - (it + 1)
        eta = remaining_iters * t_iter
        print(f"  iter {it + 1:>3}/{ACO_KWARGS['max_iter']}  "
              f"best={aco.archive_fitness[0]:9.4f}  "
              f"({t_iter:5.1f}s this iter, {elapsed / 60:5.1f}min elapsed, "
              f"~{eta / 60:5.1f}min remaining)")

    best_position = aco.archive_positions[0].copy()
    best_cost = float(aco.archive_fitness[0])

    print(f"\nACO best avg cost:                {best_cost:.4f}")
    print(f"Improvement over baseline:        {baseline_cost - best_cost:+.4f} "
          f"({100 * (baseline_cost - best_cost) / baseline_cost:.1f}%)")
    print(f"Convergence (every 10th entry):   {[round(h, 3) for h in history[::10]]}")

    # ACO_R is elitist by construction: the archive's best member can
    # only stay the same or improve each iteration, never regress.
    monotonic = all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1))
    assert monotonic, "history is not monotonically non-increasing (elitism broken)"
    print("Monotonic convergence OK (elitist archive never regresses)")

    check_feasibility(best_position, lower, upper, "best_position")
    assert best_position.shape == (27,), "vector length must be 27"

    # --- reproducibility check: same seed -> identical result (optional,
    #     off by default since it doubles total runtime -- see flag above) ---
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

    print("\nAll checks passed. `best_position` is ready to be loaded as the")
    print("final tuned controller (controller.set_params_from_vector(best_position)).")


if __name__ == "__main__":
    main()