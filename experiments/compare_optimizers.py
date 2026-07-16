"""
experiments/compare_optimizers.py

Runs PSO (optimization/pso.py) and ACO (optimization/aco.py) against the
same fuzzy-controller tuning problem and produces the four comparison
criteria required for Phase 4:

  1. Final cost-function value achieved by each algorithm
  2. Convergence speed (plotted, best-cost-so-far per iteration)
  3. Stability of each algorithm's solution across different runs
     (multiple random seeds -> mean +/- std of both final cost and
     final parameter vector)
  4. Effect of the optimization on the fuzzy controller's actual
     performance (per-scenario cost breakdown of the tuned controller,
     not just the aggregate number the optimizer minimized)

Uses the SAME robust multi-scenario fitness as run_pso_optimization.py
/ run_aco_optimization.py (4 traffic patterns x N seeds, averaged) so
this script's numbers are consistent with -- not a different
evaluation from -- those two run scripts.

Outputs:
  results/plots/pso_vs_aco_convergence.png   (criterion 2)
  results/tables/pso_vs_aco_summary.md       (criteria 1, 3, 4)
  results/tables/pso_vs_aco_summary.csv      (criterion 1, 3 -- machine-readable)
  results/tables/pso_vs_aco_result_params.md (tuned parameters, human-readable)
  results/tables/pso_vs_aco_result_params.csv

Run from the project root:
    python experiments/compare_optimizers.py
    python experiments/compare_optimizers.py --max-iter 100 --stability-seeds 1 2 3 4 5
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
from optimization.pso import PSOOptimizer, make_multi_scenario_fitness
from optimization.aco import ACOOptimizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# Same four traffic patterns used throughout Phase 2/3 (see
# src/fuzzy/README.md and run_pso_optimization.py /
# run_aco_optimization.py) -- kept identical here so this script's
# "final cost" numbers are directly comparable to those two.
SCENARIOS = [
    ("moderate symmetric",  0.3, 0.3),
    ("moderate asymmetric", 0.4, 0.2),
    ("heavy asymmetric",    0.6, 0.2),
    ("heavy symmetric",     0.5, 0.5),
]


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------

def make_env_factory(r1: float, r2: float, departure_rate: float, seed: int):
    """Returns a zero-arg callable that builds a fresh TrafficEnv.
    Defaults on r1/r2/seed avoid the classic late-binding closure bug
    when this is called inside a loop."""
    def _factory(r1=r1, r2=r2, departure_rate=departure_rate, seed=seed):
        return TrafficEnv(arrival_rate_1=r1, arrival_rate_2=r2,
                           departure_rate=departure_rate, seed=seed)
    return _factory


def build_env_factories(scenario_seeds: list[int], departure_rate: float):
    """Cross product of SCENARIOS x scenario_seeds -> env_factories.
    This is the fitness landscape both optimizers are tuned against."""
    factories, labels = [], []
    for label, r1, r2 in SCENARIOS:
        for seed in scenario_seeds:
            factories.append(make_env_factory(r1, r2, departure_rate, seed))
            labels.append(f"{label} (seed={seed})")
    return factories, labels


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


def check_feasibility(position: np.ndarray, lower: np.ndarray, upper: np.ndarray, label: str) -> None:
    assert ((position >= lower - 1e-9) & (position <= upper + 1e-9)).all(), \
        f"[{label}] position out of bounds"
    for start in range(0, 18, 3):
        a, b, c = position[start:start + 3]
        assert a <= b + 1e-9 <= c + 1e-9, f"[{label}] invalid triangle at {start}: {a},{b},{c}"


# ----------------------------------------------------------------------
# Single optimizer runs (one seed) -- used both for the "headline" run
# and for each repetition in the stability analysis
# ----------------------------------------------------------------------

def run_pso_once(fitness_fn, args: argparse.Namespace, seed: int) -> dict:
    controller = FuzzyController()
    pso = PSOOptimizer(
        controller, fitness_fn,
        num_particles=args.pso_particles,
        max_iter=args.max_iter,
        w=args.pso_w, w_min=args.pso_w_min,
        c1=args.pso_c1, c2=args.pso_c2,
        random_seed=seed,
    )
    t0 = time.time()
    best_position, best_cost, history = pso.optimize()
    elapsed = time.time() - t0
    return dict(name="PSO", seed=seed, best_position=best_position,
                best_cost=best_cost, history=history, elapsed=elapsed)


def run_aco_once(fitness_fn, args: argparse.Namespace, seed: int) -> dict:
    controller = FuzzyController()
    aco = ACOOptimizer(
        controller, fitness_fn,
        archive_size=args.aco_archive_size,
        num_ants=args.aco_num_ants,
        max_iter=args.max_iter,
        q=args.aco_q, xi=args.aco_xi,
        random_seed=seed,
    )
    t0 = time.time()
    best_position, best_cost, history = aco.optimize()
    elapsed = time.time() - t0
    return dict(name="ACO", seed=seed, best_position=best_position,
                best_cost=best_cost, history=history, elapsed=elapsed)


# ----------------------------------------------------------------------
# Criterion 3: stability across runs
# ----------------------------------------------------------------------

def stability_analysis(fitness_fn, args: argparse.Namespace) -> dict:
    """
    Re-runs both optimizers once per seed in args.stability_seeds
    (independent optimizer RNG seeds, on the SAME fitness landscape),
    and summarizes how much the final cost and final parameter vector
    vary run-to-run. Low spread = stable/reliable algorithm; high
    spread = sensitive to random initialization.
    """
    print(f"\nStability analysis: re-running each optimizer with "
          f"{len(args.stability_seeds)} different seeds "
          f"({args.stability_seeds})...")

    pso_runs, aco_runs = [], []
    for seed in args.stability_seeds:
        print(f"  [stability] PSO  seed={seed} ...", end=" ", flush=True)
        r = run_pso_once(fitness_fn, args, seed)
        pso_runs.append(r)
        print(f"cost={r['best_cost']:.4f}")

        print(f"  [stability] ACO  seed={seed} ...", end=" ", flush=True)
        r = run_aco_once(fitness_fn, args, seed)
        aco_runs.append(r)
        print(f"cost={r['best_cost']:.4f}")

    def summarize(runs: list[dict]) -> dict:
        costs = np.array([r["best_cost"] for r in runs])
        positions = np.stack([r["best_position"] for r in runs])
        # Per-dimension std across runs, then averaged, gives one
        # number summarizing how much the *parameters themselves*
        # (not just the final cost) vary run-to-run.
        param_std_mean = float(np.mean(np.std(positions, axis=0)))
        return dict(
            costs=costs.tolist(),
            cost_mean=float(np.mean(costs)),
            cost_std=float(np.std(costs)),
            cost_min=float(np.min(costs)),
            cost_max=float(np.max(costs)),
            param_std_mean=param_std_mean,
        )

    return dict(
        pso_runs=pso_runs, aco_runs=aco_runs,
        pso_summary=summarize(pso_runs), aco_summary=summarize(aco_runs),
    )


# ----------------------------------------------------------------------
# Criterion 4: effect on actual controller performance, per scenario
# ----------------------------------------------------------------------

def report_per_scenario(controller: FuzzyController, factories, labels, num_steps: int) -> list[dict]:
    """Evaluate the tuned controller on each individual scenario
    (not just the aggregate the optimizer minimized), so a good
    average can't hide a regression on one traffic pattern."""
    rows = []
    for factory, label in zip(factories, labels):
        cost = evaluate_controller(controller, factory, num_steps)
        rows.append(dict(label=label, cost=cost))
    return rows


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
    plt.ylabel("Best avg cost so far")
    plt.title("PSO vs ACO convergence (multi-scenario fitness)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_summary_report(results, baseline_cost, stability, per_scenario, out_path: Path) -> None:
    lines = ["# PSO vs ACO — comparison summary\n"]

    lines.append("## 1. Final cost-function value\n")
    lines.append(f"Baseline (default hand-picked params): **{baseline_cost:.4f}**\n")
    lines.append("| Method | Best avg cost | Improvement over baseline | Runtime (s) |")
    lines.append("|---|---|---|---|")
    for r in results:
        improvement = baseline_cost - r["best_cost"]
        pct = 100 * improvement / baseline_cost
        lines.append(f"| {r['name']} | {r['best_cost']:.4f} | "
                      f"{improvement:+.4f} ({pct:+.1f}%) | {r['elapsed']:.1f} |")

    lines.append("\n## 2. Convergence speed\n")
    lines.append("See `results/plots/pso_vs_aco_convergence.png` for the full curve. "
                  "Best-cost-so-far sampled every few iterations:\n")
    for r in results:
        step = max(1, len(r["history"]) // 8)
        sampled = [round(h, 3) for h in r["history"][::step]]
        lines.append(f"- **{r['name']}**: {sampled}")

    lines.append("\n## 3. Stability across runs\n")
    lines.append(f"Each optimizer re-run independently with seeds "
                  f"{[r['seed'] for r in stability['pso_runs']]} "
                  f"on the same multi-scenario fitness landscape.\n")
    lines.append("| Method | Cost mean | Cost std | Cost min | Cost max | "
                  "Mean per-parameter std |")
    lines.append("|---|---|---|---|---|---|")
    for name, summary in (("PSO", stability["pso_summary"]), ("ACO", stability["aco_summary"])):
        s = summary
        lines.append(f"| {name} | {s['cost_mean']:.4f} | {s['cost_std']:.4f} | "
                      f"{s['cost_min']:.4f} | {s['cost_max']:.4f} | "
                      f"{s['param_std_mean']:.4f} |")
    lines.append("\nLower `cost std` and lower `mean per-parameter std` both indicate a "
                  "more stable/reliable algorithm -- it lands on a similar answer "
                  "regardless of random seed. Higher values mean the algorithm's "
                  "result depends more on where it happened to start.\n")

    lines.append("\n## 4. Effect on fuzzy controller performance (per scenario)\n")
    lines.append("Tuned controller (best of the two optimizers by final cost) "
                  "evaluated on every individual traffic scenario/seed "
                  "it was optimized against, not just the aggregate:\n")
    lines.append("| Scenario | Cost |")
    lines.append("|---|---|")
    for row in per_scenario:
        lines.append(f"| {row['label']} | {row['cost']:.2f} |")

    out_path.write_text("\n".join(lines))


def write_summary_csv(results, baseline_cost, stability, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "best_cost", "improvement_over_baseline",
                          "runtime_s", "stability_cost_mean", "stability_cost_std",
                          "stability_param_std_mean"])
        writer.writerow(["Baseline", round(baseline_cost, 4), "", "", "", "", ""])
        for r, key in ((results[0], "pso_summary"), (results[1], "aco_summary")):
            s = stability[key]
            writer.writerow([
                r["name"], round(r["best_cost"], 4),
                round(baseline_cost - r["best_cost"], 4), round(r["elapsed"], 1),
                round(s["cost_mean"], 4), round(s["cost_std"], 4),
                round(s["param_std_mean"], 4),
            ])


def write_param_report(controller, results, baseline_cost, default_vec, out_path: Path) -> None:
    lines = ["# PSO vs ACO — tuned parameter values\n",
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


def write_param_csv(results, default_vec, out_path: Path) -> None:
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
    parser = argparse.ArgumentParser(
        description="Run PSO and ACO on the fuzzy controller and produce a "
                    "full Phase 4 comparison (final cost, convergence, "
                    "stability, per-scenario effect)."
    )

    # Simulation / cost -- NOTE: decision_interval intentionally removed.
    # evaluate_controller() no longer accepts it (Phase 2's block-based
    # switching fix); this was a leftover from before that fix and would
    # raise TypeError if kept.
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--departure-rate", type=float, default=1.0)
    parser.add_argument("--scenario-seeds", type=int, nargs="+", default=[1, 2, 3],
                         help="Seeds used to build the multi-scenario fitness "
                              "landscape both optimizers are tuned against "
                              "(4 traffic patterns x these seeds).")

    # Shared optimizer settings
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7,
                         help="Seed for the headline PSO/ACO run (the one "
                              "used for the convergence plot and per-scenario "
                              "report).")
    parser.add_argument("--stability-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                         help="Independent optimizer seeds used for the "
                              "stability analysis (criterion 3).")

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

    parser.add_argument("--skip-stability", action="store_true",
                         help="Skip the stability re-runs (criterion 3) to "
                              "save time, e.g. for a quick sanity check.")
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

    env_factories, labels = build_env_factories(args.scenario_seeds, args.departure_rate)
    print(f"Fitness landscape: {len(SCENARIOS)} traffic patterns x "
          f"{len(args.scenario_seeds)} seeds = {len(env_factories)} scenario/seed "
          f"combinations, averaged per candidate.\n")

    controller = FuzzyController()
    lower, upper = controller.get_param_bounds()

    fitness_fn = make_multi_scenario_fitness(
        env_factories=env_factories, num_steps=args.num_steps, aggregate=np.mean,
    )

    # --- baseline ---
    default_vec = controller.get_default_vector()
    controller.set_params_from_vector(default_vec)
    baseline_cost = fitness_fn(controller)
    print(f"Baseline (default vector) avg cost: {baseline_cost:.4f}")

    # --- headline PSO / ACO runs (used for plot + per-scenario report) ---
    print("\nRunning headline PSO run (seed={})...".format(args.seed))
    pso_result = run_pso_once(fitness_fn, args, args.seed)
    print(f"PSO best avg cost: {pso_result['best_cost']:.4f}  ({pso_result['elapsed']:.1f}s)")
    check_feasibility(pso_result["best_position"], lower, upper, "PSO best_position")

    print("\nRunning headline ACO run (seed={})...".format(args.seed))
    aco_result = run_aco_once(fitness_fn, args, args.seed)
    print(f"ACO best avg cost: {aco_result['best_cost']:.4f}  ({aco_result['elapsed']:.1f}s)")
    check_feasibility(aco_result["best_position"], lower, upper, "ACO best_position")

    results = [pso_result, aco_result]

    # --- criterion 2: convergence plot ---
    plot_path = plots_dir / "pso_vs_aco_convergence.png"
    plot_convergence(results, baseline_cost, plot_path)
    print(f"\nSaved convergence plot to {plot_path}")

    # --- criterion 3: stability across independent runs ---
    if args.skip_stability:
        print("\nSkipping stability analysis (--skip-stability set).")
        stability = dict(
            pso_runs=[pso_result], aco_runs=[aco_result],
            pso_summary=dict(costs=[pso_result["best_cost"]], cost_mean=pso_result["best_cost"],
                              cost_std=0.0, cost_min=pso_result["best_cost"],
                              cost_max=pso_result["best_cost"], param_std_mean=0.0),
            aco_summary=dict(costs=[aco_result["best_cost"]], cost_mean=aco_result["best_cost"],
                              cost_std=0.0, cost_min=aco_result["best_cost"],
                              cost_max=aco_result["best_cost"], param_std_mean=0.0),
        )
    else:
        stability = stability_analysis(fitness_fn, args)
        print(f"\nPSO stability: cost {stability['pso_summary']['cost_mean']:.4f} "
              f"+/- {stability['pso_summary']['cost_std']:.4f}")
        print(f"ACO stability: cost {stability['aco_summary']['cost_mean']:.4f} "
              f"+/- {stability['aco_summary']['cost_std']:.4f}")

    # --- criterion 4: effect on actual controller performance ---
    best_of_two = min(results, key=lambda r: r["best_cost"])
    controller.set_params_from_vector(best_of_two["best_position"])
    per_scenario = report_per_scenario(controller, env_factories, labels, args.num_steps)
    print(f"\nPer-scenario breakdown computed for the better of the two "
          f"tuned controllers ({best_of_two['name']}).")

    # --- write reports ---
    summary_md = tables_dir / "pso_vs_aco_summary.md"
    write_summary_report(results, baseline_cost, stability, per_scenario, summary_md)
    print(f"Saved comparison summary to {summary_md}")

    summary_csv = tables_dir / "pso_vs_aco_summary.csv"
    write_summary_csv(results, baseline_cost, stability, summary_csv)
    print(f"Saved comparison summary (CSV) to {summary_csv}")

    param_md = tables_dir / "pso_vs_aco_result_params.md"
    write_param_report(controller, results, baseline_cost, default_vec, param_md)
    print(f"Saved parameter report to {param_md}")

    param_csv = tables_dir / "pso_vs_aco_result_params.csv"
    write_param_csv(results, default_vec, param_csv)
    print(f"Saved parameter table (CSV) to {param_csv}")

    # --- console summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<10}{'Best cost':>12}{'Improvement':>15}{'Runtime (s)':>14}")
    print(f"{'Baseline':<10}{baseline_cost:>12.4f}{'--':>15}{'--':>14}")
    for r in results:
        print(f"{r['name']:<10}{r['best_cost']:>12.4f}"
              f"{baseline_cost - r['best_cost']:>+15.4f}{r['elapsed']:>14.1f}")


if __name__ == "__main__":
    main()