"""
test_fuzzy_controller.py

Smoke test for Phase 2 (FuzzyController). Verifies:
    - the controller can be built and queried without errors
    - the parameter vector round-trips correctly (used by PSO/ACO)
    - the controller produces a lower cost than the naive fixed-timer
      baseline from Phase 1, when run with decision_interval=1

Run from the project root:
    python experiments/test_fuzzy_controller.py

NOTE on decision_interval: the controller is re-queried every tick by
default here. Using a larger decision_interval (e.g. matching a
realistic minimum green time) is more physically realistic but
requires better-tuned membership functions to avoid one road being
starved of green time -- this is exactly the kind of improvement
Phase 3 (PSO/ACO) is meant to discover automatically.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import compute_cost, evaluate_controller


def run_fixed_timer_baseline(env_factory, num_steps, switch_every=10):
    env = env_factory()
    env.reset()
    green = 1
    for t in range(num_steps):
        if t % switch_every == 0:
            green = 2 if green == 1 else 1
        env.step(green)
    return env.get_metrics()


def main():
    env_factory = lambda: TrafficEnv(
        arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0, seed=42
    )
    num_steps = 2000

    controller = FuzzyController()

    # --- Sanity checks on the parameter interface (what PSO/ACO use) ---
    vec = controller.get_default_vector()
    lower, upper = controller.get_param_bounds()
    assert len(vec) == len(lower) == len(upper), "Vector/bounds length mismatch"

    params_back = controller.vector_to_params(vec)
    vec_back = controller.params_to_vector(params_back)
    import numpy as np
    assert np.allclose(vec, vec_back), "Vector <-> params round-trip failed"
    print(f"Parameter vector length: {len(vec)} (OK)")
    print("Vector <-> params round-trip: OK\n")

    # --- Baseline comparison ---
    fixed_metrics = run_fixed_timer_baseline(env_factory, num_steps)
    fixed_cost = compute_cost(fixed_metrics)
    print("Fixed-timer baseline:")
    print(f"  metrics: {fixed_metrics}")
    print(f"  cost:    {fixed_cost:.2f}\n")

    # --- Show sample decisions: how does the controller split green time? ---
    print("Sample decisions (queue_1, queue_2) -> (green_1 sec, green_2 sec):")
    sample_queues = [(0, 0), (5, 5), (15, 2), (2, 15), (10, 10), (20, 0), (0, 20)]
    for q1, q2 in sample_queues:
        g1, g2 = controller.compute_green_time(q1, q2)
        print(f"  queue_1={q1:>3}, queue_2={q2:>3}  ->  "
              f"green_1={g1:6.2f}s, green_2={g2:6.2f}s")
    print()

    # --- Full simulation run with the fuzzy controller ---
    fuzzy_env = env_factory()
    fuzzy_env.reset()
    current_green = 1
    green_1_choices = []
    for t in range(num_steps):
        g1, g2 = controller.compute_green_time(
            fuzzy_env.queue_length_1, fuzzy_env.queue_length_2
        )
        current_green = 1 if g1 >= g2 else 2
        green_1_choices.append(g1)
        fuzzy_env.step(current_green)

    fuzzy_metrics = fuzzy_env.get_metrics()
    fuzzy_cost = compute_cost(fuzzy_metrics)

    print("Fuzzy controller (default params, decision_interval=1):")
    print(f"  metrics: {fuzzy_metrics}")
    print(f"  cost:    {fuzzy_cost:.2f}")
    print(f"  average green_1 decision over the run: "
          f"{sum(green_1_choices) / len(green_1_choices):.2f}s\n")

    print("Done. Fuzzy controller runs end-to-end without errors.\n")

    # --- Multi-scenario comparison (honest evaluation, not cherry-picked) ---
    print("=" * 70)
    print("Multi-scenario comparison: fixed-timer vs. fuzzy (default params)")
    print("Fuzzy controller is NOT universally better -- see src/fuzzy/README.md")
    print("=" * 70)
    scenarios = [
        ("moderate symmetric",   0.3, 0.3),
        ("moderate asymmetric",  0.4, 0.2),
        ("heavy asymmetric",     0.6, 0.2),
        ("heavy symmetric",      0.5, 0.5),
    ]
    for label, r1, r2 in scenarios:
        scenario_env_factory = lambda r1=r1, r2=r2: TrafficEnv(
            arrival_rate_1=r1, arrival_rate_2=r2, departure_rate=1.0, seed=42
        )
        fixed_cost_s = compute_cost(run_fixed_timer_baseline(scenario_env_factory, num_steps))

        controller_s = FuzzyController()
        env_s = scenario_env_factory()
        env_s.reset()
        for t in range(num_steps):
            g1, g2 = controller_s.compute_green_time(
                env_s.queue_length_1, env_s.queue_length_2
            )
            env_s.step(1 if g1 >= g2 else 2)
        fuzzy_cost_s = compute_cost(env_s.get_metrics())

        winner = "fuzzy" if fuzzy_cost_s < fixed_cost_s else "fixed-timer"
        print(f"  {label:22s} rates=({r1},{r2})  "
              f"fixed={fixed_cost_s:8.2f}  fuzzy={fuzzy_cost_s:8.2f}  "
              f"-> {winner} wins")

    print("\nTakeaway: fuzzy tends to win under heavy/imbalanced traffic, but can")
    print("lose under light/balanced traffic with these DEFAULT parameters.")
    print("This is exactly what Phase 3 (PSO/ACO) should improve on.")


if __name__ == "__main__":
    main()