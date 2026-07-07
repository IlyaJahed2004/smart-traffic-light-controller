"""
test_traffic_env.py

Quick smoke test for Phase 1 (TrafficEnv). Not part of the final
experiments/results -- just a sanity check that the simulation runs
without errors and produces sensible metrics.

Run from the project root:
    python experiments/test_traffic_env.py
"""

import sys
import os

# Allow running this script directly without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulation.traffic_env import TrafficEnv


def main():
    env = TrafficEnv(
        arrival_rate_1=0.4,
        arrival_rate_2=0.2,
        departure_rate=1.0,
        seed=42,
    )
    env.reset()

    # Placeholder fixed-timer controller: alternate every 10 ticks.
    # This will be replaced by the fuzzy controller in Phase 2.
    green = 1
    num_steps = 2000
    for t in range(num_steps):
        if t % 10 == 0:
            green = 2 if green == 1 else 1
        env.step(green)

    metrics = env.get_metrics()

    print(f"Ran {num_steps} simulation steps.\n")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"\nFinal queue lengths -> road_1: {env.queue_length_1}, "
          f"road_2: {env.queue_length_2}")


if __name__ == "__main__":
    main()