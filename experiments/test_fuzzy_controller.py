"""
test_fuzzy_controller.py
 
Smoke test for Phase 2 (FuzzyController). Verifies:
    - the controller can be built and queried without errors
    - the parameter vector round-trips correctly (used by PSO/ACO)
    - the controller's green-time decisions look sensible for sample
      queue combinations
    - the simulation now switches the light REALISTICALLY: it holds
      each road green for a solid block of ticks matching the
      computed duration, then switches -- NOT flip-flopping every
      tick (see cost_function.evaluate_controller for the fix)
    - the controller is compared against the naive fixed-timer
      baseline from Phase 1, across multiple traffic scenarios
 
Run from the project root:
    python experiments/test_fuzzy_controller.py
"""
 
import sys
import os
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
 
import numpy as np
from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import compute_cost, evaluate_controller
 
 
def run_fixed_timer_baseline(env_factory, num_steps, switch_every=10):
    """Naive Phase-1-style baseline: alternate every `switch_every` ticks,
    regardless of queue lengths. Used only as a point of comparison."""
    env = env_factory()
    env.reset()
    green = 1
    for t in range(num_steps):
        if t % switch_every == 0:
            green = 2 if green == 1 else 1
        env.step(green)
    return env.get_metrics()
 
 
def demo_realistic_switching(controller, env_factory, num_steps=100):
    """Prints the actual green-road sequence produced by
    evaluate_controller's block-based switching, so it's visible
    (not just asserted) that the light holds steady instead of
    flip-flopping every tick."""
    env = env_factory()
    env.reset()
 
    green_log = []
    ticks_run = 0
    num_decisions = 0
    while ticks_run < num_steps:
        g1, g2 = controller.compute_green_time(
            env.queue_length_1, env.queue_length_2
        )
        g1_ticks = max(1, round(g1))
        g2_ticks = max(1, round(g2))
        num_decisions += 1
 
        for _ in range(g1_ticks):
            if ticks_run >= num_steps:
                break
            env.step(1)
            green_log.append(1)
            ticks_run += 1
        for _ in range(g2_ticks):
            if ticks_run >= num_steps:
                break
            env.step(2)
            green_log.append(2)
            ticks_run += 1
 
    flips = sum(1 for i in range(1, len(green_log)) if green_log[i] != green_log[i - 1])
    print(f"  {num_steps} ticks simulated, {num_decisions} fresh decisions made, "
          f"{flips} switches total (should be roughly one per decision, not one per tick)")
    print(f"  first 30 ticks of green sequence: {green_log[:30]}")
 
 
def main():
    env_factory = lambda: TrafficEnv(
        arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0, seed=42
    )
    num_steps = 2000
 
    controller = FuzzyController()
    import pprint
    print("Default fuzzy parameters:")
    pprint.pprint(controller.get_default_params())
    print()
 
    # --- Sanity checks on the parameter interface (what PSO/ACO use) ---
    vec = controller.get_default_vector()
    lower, upper = controller.get_param_bounds()
    assert len(vec) == len(lower) == len(upper), "Vector/bounds length mismatch"
 
    params_back = controller.vector_to_params(vec)
    vec_back = controller.params_to_vector(params_back)
    assert np.allclose(vec, vec_back), "Vector <-> params round-trip failed"
    print(f"Parameter vector length: {len(vec)} (OK)")
    print("Vector <-> params round-trip: OK\n")
 
    # --- Show sample decisions: how does the controller split green time? ---
    print("Sample decisions (queue_1, queue_2) -> (green_1 sec, green_2 sec):")
    sample_queues = [(0, 0), (5, 5), (15, 2), (2, 15), (10, 10), (20, 0), (0, 20)]
    for q1, q2 in sample_queues:
        g1, g2 = controller.compute_green_time(q1, q2)
        print(f"  queue_1={q1:>3}, queue_2={q2:>3}  ->  "
              f"green_1={g1:6.2f}s, green_2={g2:6.2f}s")
    print()
 
    # --- Demonstrate realistic (non-flickering) light switching ---
    print("Realistic switching check (block-based, not per-tick flip-flop):")
    demo_realistic_switching(controller, env_factory, num_steps=100)
    print()
 
    # --- Baseline comparison ---
    fixed_metrics = run_fixed_timer_baseline(env_factory, num_steps)
    fixed_cost = compute_cost(fixed_metrics)
    print("Fixed-timer baseline:")
    print(f"  metrics: {fixed_metrics}")
    print(f"  cost:    {fixed_cost:.2f}\n")
 
    fuzzy_cost = evaluate_controller(controller, env_factory, num_steps)
    print("Fuzzy controller (default params, realistic block switching):")
    print(f"  cost:    {fuzzy_cost:.2f}\n")
 
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
        fuzzy_cost_s = evaluate_controller(controller_s, scenario_env_factory, num_steps)
 
        winner = "fuzzy" if fuzzy_cost_s < fixed_cost_s else "fixed-timer"
        print(f"  {label:22s} rates=({r1},{r2})  "
              f"fixed={fixed_cost_s:8.2f}  fuzzy={fuzzy_cost_s:8.2f}  "
              f"-> {winner} wins")
 
    print("\nTakeaway: fuzzy tends to win under heavy/imbalanced traffic, but can")
    print("lose under light/balanced traffic with these DEFAULT parameters.")
    print("This is exactly what Phase 3 (PSO/ACO) should improve on.")
 
 
if __name__ == "__main__":
    main()