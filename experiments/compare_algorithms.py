"""
Runs PSO (optimization/pso.py) and ACO (optimization/aco.py) against the
same fuzzy-controller tuning problem, then:

  1. Plots both convergence curves (+ baseline) on one chart, saved to
     results/plots/.
  2. Writes each optimizer's best 27-parameter vector back into
     human-readable form (membership-function points + named rules),
     saved as both a Markdown report and a CSV table to results/tables/.

Mirrors the setup used in test_pso_smoke.py / test_aco_smoke.py, but
exposes every knob via CLI flags so it can be run as a quick smoke
comparison or scaled up into a real experiment.

Run from the project root:
    python experiments/compare_optimizers.py
    python experiments/compare_optimizers.py --num-steps 1000 --max-iter 100
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fuzzy.fuzzy_controller import FuzzyController
from simulation.traffic_env import TrafficEnv
from cost_function import evaluate_controller
from optimization.pso import PSOOptimizer, make_single_scenario_fitness
from optimization.aco import ACOOptimizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def make_env_factory(args: argparse.Namespace):
    return lambda: TrafficEnv(
        arrival_rate_1=args.arrival_rate_1,
        arrival_rate_2=args.arrival_rate_2,
        departure_rate=args.departure_rate,
        seed=args.env_seed,
    )


def flat_param_names() -> list[str]:
    """Names for each of the 27 vector entries, in the exact order
    produced by FuzzyController.params_to_vector."""
    names: list[str] = []
    for set_name in FuzzyController.INPUT_SETS:
        for point in ("a", "b", "c"):
            names.append(f"mf_queue.{set_name}.{point}")
    for set_name in FuzzyController.OUTPUT_SETS:
        for point in ("a", "b", "c"):
            names.append(f"mf_green.{set_name}.{point}")
    for i, (s1, s2, out) in enumerate(FuzzyController.RULES):
        names.append(f"rule_weights[{i}] (IF q1={s1} AND q2={s2} THEN green={out})")
    return names


def vector_to_readable(controller: FuzzyController, vector: np.ndarray) -> dict:
    """Same information as vector_to_params, but with rule weights
    keyed by a human-readable rule description instead of a bare index."""
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


# ----------------------------------------------------------------------
# Optimizer runs
# ----------------------------------------------------------------------

def run_pso(controller, fitness_fn, args: argparse.Namespace) -> dict:
    pso = PSOOptimizer(
        controller, fitness_fn,
        num_particles=args.pso_particles,
        max_iter=args.max_iter,
        w=args.pso_w, w_min=args.pso_w_min,
        c1=args.pso_c1, c2=args.pso_c2,
        random_seed=args.seed,
    )
    t0 = time.time()
    best_position, best_cost, history = pso.optimize()
    elapsed = time.time() - t0
    return dict(name="PSO", best_position=best_position, best_cost=best_cost,
                history=history, elapsed=elapsed)


def run_aco(controller, fitness_fn, args: argparse.Namespace) -> dict:
    aco = ACOOptimizer(
        controller, fitness_fn,
        archive_size=args.aco_archive_size,
        num_ants=args.aco_num_ants,
        max_iter=args.max_iter,
        q=args.aco_q, xi=args.aco_xi,
        random_seed=args.seed,
    )
    t0 = time.time()
    best_position, best_cost, history = aco.optimize()
    elapsed = time.time() - t0
    return dict(name="ACO", best_position=best_position, best_cost=best_cost,
                history=history, elapsed=elapsed)


# ----------------------------------------------------------------------
# Output: plot + tables
# ----------------------------------------------------------------------

def plot_convergence(results: list[dict], baseline_cost: float, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(r["history"], label=f'{r["name"]} (best={r["best_cost"]:.3f})')
    plt.axhline(baseline_cost, color="gray", linestyle="--",
                label=f"Baseline ({baseline_cost:.3f})")
    plt.xlabel("Iteration")
    plt.ylabel("Best cost so far")
    plt.title("PSO vs ACO convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_markdown_report(controller, results, baseline_cost, default_vec, out_path: Path) -> None:
    lines = ["# PSO vs ACO — result parameters\n",
             f"Baseline cost (default vector): **{baseline_cost:.4f}**\n",
             "\n## Baseline\n```json",
             json.dumps(vector_to_readable(controller, default_vec), indent=2),
             "```\n"]

    for r in results:
        lines.append(f"## {r['name']}\n")
        lines.append(f"- Best cost: **{r['best_cost']:.4f}**")
        lines.append(f"- Improvement over baseline: **{baseline_cost - r['best_cost']:+.4f}**")
        lines.append(f"- Runtime: {r['elapsed']:.1f}s\n")
        lines.append("```json")
        lines.append(json.dumps(vector_to_readable(controller, r["best_position"]), indent=2))
        lines.append("```\n")

    out_path.write_text("\n".join(lines))


def write_csv_table(results, default_vec, out_path: Path) -> None:
    names = flat_param_names()
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["param_name", "baseline"] + [r["name"] for r in results]
        writer.writerow(header)
        columns = [default_vec] + [r["best_position"] for r in results]
        for i, name in enumerate(names):
            writer.writerow([name] + [round(float(col[i]), 4) for col in columns])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSO and ACO and compare convergence.")

    # Simulation / cost
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--arrival-rate-1", type=float, default=0.4)
    parser.add_argument("--arrival-rate-2", type=float, default=0.2)
    parser.add_argument("--departure-rate", type=float, default=1.0)
    parser.add_argument("--env-seed", type=int, default=42)

    # Shared optimizer settings
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)

    # PSO-specific
    parser.add_argument("--pso-particles", type=int, default=30)
    parser.add_argument("--pso-w", type=float, default=0.7)
    parser.add_argument("--pso-w-min", type=float, default=0.4)
    parser.add_argument("--pso-c1", type=float, default=1.5)
    parser.add_argument("--pso-c2", type=float, default=1.5)

    # ACO-specific
    parser.add_argument("--aco-archive-size", type=int, default=20)
    parser.add_argument("--aco-num-ants", type=int, default=10)
    parser.add_argument("--aco-q", type=float, default=0.3)
    parser.add_argument("--aco-xi", type=float, default=0.85)

    parser.add_argument("--output-dir", type=str, default=None,
                         help="Override the project's results/ directory "
                              "(plots/ and tables/ subfolders are created under it)")
    args = parser.parse_args()

    if args.output_dir:
        plots_dir = Path(args.output_dir) / "plots"
        tables_dir = Path(args.output_dir) / "tables"
    else:
        plots_dir, tables_dir = DEFAULT_PLOTS_DIR, DEFAULT_TABLES_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    controller = FuzzyController()
    env_factory = make_env_factory(args)

    # --- baseline ---
    default_vec = controller.get_default_vector()
    controller.set_params_from_vector(default_vec)
    baseline_cost = evaluate_controller(
        controller, env_factory,
        num_steps=args.num_steps, decision_interval=args.decision_interval,
    )
    print(f"Baseline (default vector) cost: {baseline_cost:.4f}")

    fitness_fn = make_single_scenario_fitness(
        env_factory, num_steps=args.num_steps, decision_interval=args.decision_interval,
    )

    # --- PSO ---
    print("\nRunning PSO...")
    pso_result = run_pso(controller, fitness_fn, args)
    print(f"PSO best cost: {pso_result['best_cost']:.4f}  ({pso_result['elapsed']:.1f}s)")

    # --- ACO ---
    print("\nRunning ACO...")
    aco_result = run_aco(controller, fitness_fn, args)
    print(f"ACO best cost: {aco_result['best_cost']:.4f}  ({aco_result['elapsed']:.1f}s)")

    results = [pso_result, aco_result]

    # --- outputs ---
    plot_path = plots_dir / "pso_vs_aco_convergence.png"
    plot_convergence(results, baseline_cost, plot_path)
    print(f"\nSaved convergence plot to {plot_path}")

    md_path = tables_dir / "pso_vs_aco_result_params.md"
    write_markdown_report(controller, results, baseline_cost, default_vec, md_path)
    print(f"Saved parameter report to {md_path}")

    csv_path = tables_dir / "pso_vs_aco_result_params.csv"
    write_csv_table(results, default_vec, csv_path)
    print(f"Saved parameter table (CSV) to {csv_path}")

    # --- summary ---
    print("\nSummary:")
    print(f"{'Method':<10}{'Best cost':>12}{'Improvement':>15}{'Runtime (s)':>14}")
    print(f"{'Baseline':<10}{baseline_cost:>12.4f}{'--':>15}{'--':>14}")
    for r in results:
        print(f"{r['name']:<10}{r['best_cost']:>12.4f}"
              f"{baseline_cost - r['best_cost']:>+15.4f}{r['elapsed']:>14.1f}")


if __name__ == "__main__":
    main()