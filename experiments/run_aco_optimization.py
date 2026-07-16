"""
test_aco_smoke.py

Quick, cheap sanity check for optimization/aco.py. Mirrors
test_pso_smoke.py so the two algorithms can be checked (and later
compared) the same way.

Run from the project root:
    python test_aco_smoke.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import evaluate_controller
from optimization.aco import ACOOptimizer, make_single_scenario_fitness

def make_env_factory():
    return lambda: TrafficEnv(
        arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0, seed=42
    )


def check_feasibility(position, lower, upper, label):
    assert ((position >= lower - 1e-9) & (position <= upper + 1e-9)).all(), \
        f"[{label}] position out of bounds"
    for start in range(0, 18, 3):
        a, b, c = position[start:start + 3]
        assert a <= b + 1e-9 <= c + 1e-9, f"[{label}] invalid triangle at {start}: {a},{b},{c}"
    print(f"[{label}] OK: within bounds, all triangles valid")


def main():
    controller = FuzzyController()
    lower, upper = controller.get_param_bounds()
    env_factory = make_env_factory()

    # --- baseline cost, for comparison ---
    default_vec = controller.get_default_vector()
    controller.set_params_from_vector(default_vec)
    baseline_cost = evaluate_controller(controller, env_factory, num_steps=300)
    print(f"Baseline (default vector) cost: {baseline_cost:.4f}")

    # --- small, fast ACO run ---
    fitness_fn = make_single_scenario_fitness(env_factory, num_steps=300)
    aco = ACOOptimizer(
        controller, fitness_fn,
        archive_size=15, num_ants=10, max_iter=20,
        q=0.3, xi=0.85,
        random_seed=7,
    )
    best_position, best_cost, history = aco.optimize()

    print(f"ACO best cost:                   {best_cost:.4f}")
    print(f"Improvement over baseline:        {baseline_cost - best_cost:+.4f}")
    print(f"Convergence (every 4th entry):    {[round(h, 3) for h in history[::4]]}")

    # ACO_R is elitist by construction: the archive's best member can
    # only stay the same or improve each iteration, never regress.
    monotonic = all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1))
    assert monotonic, "history is not monotonically non-increasing (elitism broken)"
    print("Monotonic convergence OK (elitist archive never regresses)")

    check_feasibility(best_position, lower, upper, "best_position")
    assert best_position.shape == (27,), "vector length must be 27"

    # --- reproducibility check: same seed -> identical result ---
    aco_repeat = ACOOptimizer(
        controller, fitness_fn,
        archive_size=15, num_ants=10, max_iter=20,
        q=0.3, xi=0.85,
        random_seed=7,
    )
    best_position_2, best_cost_2, _ = aco_repeat.optimize()
    assert best_cost == best_cost_2, "same seed produced different cost"
    assert np.allclose(best_position, best_position_2), "same seed produced different position"
    print("Reproducibility OK: identical seed -> identical result")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()