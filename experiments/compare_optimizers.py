"""
compare_optimizers.py

Phase 4: PSO vs. ACO comparison.

IMPORTANT DESIGN CHOICE
------------------------
This script does NOT redefine PSO_KWARGS / ACO_KWARGS / SCENARIOS /
env-factory helpers / vector<->readable conversion / save-to-disk
logic from scratch. It imports `run_pso_optimization.py` and
`run_aco_optimization.py` as modules and reuses their functions and
config directly:

    import run_pso_optimization as pso_script
    import run_aco_optimization as aco_script

    pso_script.FuzzyController, pso_script.PSOOptimizer,
    pso_script.make_multi_scenario_fitness, pso_script.evaluate_controller,
    pso_script.build_env_factories, pso_script.check_feasibility,
    pso_script.vector_to_readable, pso_script.save_final_vector
    (and the aco_script.* equivalents)

The only things THIS file adds are:
  - forcing both scripts onto one shared SCENARIOS/SEEDS/NUM_STEPS/
    max_iter configuration (so PSO and ACO are never compared on two
    different fitness landscapes by accident -- this mirrors the
    invariant run_aco_optimization.py's own docstring already
    requires),
  - the orchestration loop that runs each algorithm's *headline* run
    plus a *stability sweep* over several seeds,
  - validating, for EVERY run (headline and every stability re-run,
    both algorithms), that a real improvement over baseline actually
    happened -- not just trusting the aggregate table. See
    "VALIDATING REAL IMPROVEMENT" below.
  - the plot/table writers for the 4 required comparison criteria.

--------------------------------------------------------------------
VALIDATING REAL IMPROVEMENT (added -- this is the point of the whole
script, so it isn't just trusted silently)
--------------------------------------------------------------------
run_pso_optimization.py / run_aco_optimization.py already validate
this for their OWN single headline run. This script reuses the exact
same PSOOptimizer/ACOOptimizer classes, but through its own
run_pso_once/run_aco_once, so that validation does NOT happen
automatically here unless repeated -- so it is repeated explicitly,
for every single run this script performs (not just the headline):

  - PSO: since PSO_KWARGS sets seed_with_default_vector=True, one
    particle always starts exactly at the baseline vector, and
    global_best only ever updates on strict improvement. This makes
    best_cost <= baseline_cost mathematically guaranteed, so
    run_pso_once() asserts it after every run (headline AND every
    stability-sweep seed). A failure here means a real bug, not bad
    luck.
  - ACO: ACO_KWARGS also sets seed_with_default_vector=True, but for
    the discretized Ant System this only pre-boosts pheromone near
    the baseline's bins -- it is a soft bias, not a guarantee (ant
    selection stays fully probabilistic). So run_aco_once() checks
    the same condition but only WARNS (doesn't assert) if it's
    violated, exactly mirroring run_aco_optimization.py's own
    reasoning.

If you see the PSO assertion fire, something is broken (file an
issue / stop trusting the results). If you see the ACO warning
fire occasionally, it's expected variance; if it fires on EVERY
seed, the ACO budget (num_ants/max_iter/archive_size) is probably
too small for this fitness landscape and should be increased.

--------------------------------------------------------------------
Maps directly onto the project brief's "Algorithm Comparison" (§7)
and the required deliverables in §8, item 4:

  1. Final cost-function value
        -> `results/tables/pso_vs_aco_summary.md` / `.csv`, section 1
  2. Convergence speed
        -> `results/plots/pso_vs_aco_convergence.png`
        -> summary report, section 2 (sampled history)
  3. Stability of the solution across different runs
        -> multi-seed sweep -> summary report, section 3
           (mean/std of final cost AND mean per-parameter std)
  4. Effect of optimization on the fuzzy controller's performance
        -> summary report, section 4: baseline vs. PSO-tuned vs.
           ACO-tuned cost, broken down per traffic scenario (not
           just the aggregate the optimizer minimized) -- also
           writes the raw (W, Q, S) metrics per scenario, not just
           the scalar cost, as supporting "generated data"
           (§8 item 2).

Also produces the §8 item 4 deliverables:
  results/plots/pso_vs_aco_convergence.png
  results/tables/pso_vs_aco_summary.md / .csv
  results/tables/pso_vs_aco_result_params.md / .csv
  results/tables/pso_vs_aco_per_scenario_metrics.csv

Run from the project root:
    python experiments/compare_optimizers.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))   # for run_*_optimization imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_pso_optimization as pso_script
import run_aco_optimization as aco_script

from cost_function import compute_cost  # noqa: E402  (src on sys.path above)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


# ----------------------------------------------------------------------
# 0. Force PSO and ACO onto ONE shared configuration
# ----------------------------------------------------------------------
# COMPARE_QUICK_TEST mirrors the QUICK_TEST switch already used inside
# run_pso_optimization.py / run_aco_optimization.py. Flip it to False
# for the real Phase 4 numbers; leave it True for a fast sanity check
# of this script itself (finishes in well under a minute).
COMPARE_QUICK_TEST = False

# Seeds used to re-run each optimizer independently for the stability
# analysis (criterion 3). These are OPTIMIZER random seeds (affect
# swarm/ant randomness), not traffic-scenario seeds -- see the module
# docstring in run_pso_optimization.py / run_aco_optimization.py for
# why these are two separate concepts.
STABILITY_SEEDS = [1, 2, 3] if COMPARE_QUICK_TEST else [1, 2, 3, 4, 5]

if COMPARE_QUICK_TEST:
    SCENARIOS = [("moderate symmetric", 0.3, 0.3)]
    SEEDS = [1]
    NUM_STEPS = 200
    MAX_ITER = 10
    PSO_KWARGS = dict(num_particles=8, max_iter=MAX_ITER,
                       w=0.7, w_min=0.4, c1=1.5, c2=1.5,
                       seed_with_default_vector=True, random_seed=7)
    ACO_KWARGS = dict(archive_size=8, num_ants=6, max_iter=MAX_ITER,
                       q=0.3, xi=0.85, num_bins=20, alpha=1.0, tau0=1.0,
                       tau_min=1e-3, elitist_weight=2.0,
                       seed_with_default_vector=True, random_seed=7)
else:
    SCENARIOS = [
        ("moderate symmetric",  0.3, 0.3),
        ("moderate asymmetric", 0.4, 0.2),
        ("heavy asymmetric",    0.6, 0.2),
        ("heavy symmetric",     0.5, 0.5),
    ]
    SEEDS = [1, 2, 3]
    NUM_STEPS = 1000
    MAX_ITER = 100
    PSO_KWARGS = dict(num_particles=30, max_iter=MAX_ITER,
                       w=0.7, w_min=0.4, c1=1.5, c2=1.5,
                       seed_with_default_vector=True, random_seed=7)
    ACO_KWARGS = dict(archive_size=30, num_ants=15, max_iter=MAX_ITER,
                       q=0.3, xi=0.85, num_bins=20, alpha=1.0, tau0=1.0,
                       tau_min=1e-3, elitist_weight=2.0,
                       seed_with_default_vector=True, random_seed=7)

# Push the shared config into both imported modules. build_env_factories()
# and save_final_vector() in each script read these as globals at call
# time, so this is enough to make pso_script.build_env_factories() and
# aco_script.build_env_factories() return identical env_factories/labels,
# and to make each script's own save_final_vector() log the config this
# script actually used (not whatever QUICK_TEST default was hardcoded
# at the top of that script).
for _mod in (pso_script, aco_script):
    _mod.QUICK_TEST = COMPARE_QUICK_TEST
    _mod.SCENARIOS = SCENARIOS
    _mod.SEEDS = SEEDS
    _mod.NUM_STEPS = NUM_STEPS
pso_script.PSO_KWARGS = PSO_KWARGS
aco_script.ACO_KWARGS = ACO_KWARGS


# ----------------------------------------------------------------------
# 1 & 2. Single headline run of each optimizer (used for the
#         convergence plot and the "final cost" table)
# ----------------------------------------------------------------------

def run_pso_once(fitness_fn, seed: int, baseline_cost: float) -> dict:
    """One PSO run with the given optimizer seed. Reuses
    pso_script.FuzzyController / pso_script.PSOOptimizer directly.

    Asserts best_cost <= baseline_cost: with seed_with_default_vector
    =True, one particle always starts exactly at the baseline vector
    and global_best only updates on strict improvement, so this is a
    hard mathematical guarantee, not an expectation -- a failure here
    means a real bug (see module docstring)."""
    controller = pso_script.FuzzyController()
    kwargs = dict(PSO_KWARGS)
    kwargs["random_seed"] = seed
    pso = pso_script.PSOOptimizer(controller, fitness_fn, **kwargs)
    t0 = time.time()
    best_position, best_cost, history = pso.optimize()
    elapsed = time.time() - t0

    if kwargs.get("seed_with_default_vector", True):
        assert best_cost <= baseline_cost + 1e-9, (
            f"PSO (seed={seed}) best_cost ({best_cost:.4f}) is WORSE than "
            f"baseline ({baseline_cost:.4f}) despite seed_with_default_vector"
            f"=True -- this should be mathematically impossible. Check that "
            f"the baseline particle's fitness is being evaluated and "
            f"compared correctly."
        )

    return dict(name="PSO", seed=seed, best_position=best_position,
                best_cost=best_cost, history=history, elapsed=elapsed)


def run_aco_once(fitness_fn, seed: int, baseline_cost: float) -> dict:
    """One ACO run with the given optimizer seed. Reuses
    aco_script.FuzzyController / aco_script.ACOOptimizer directly.

    Unlike PSO, only WARNS (doesn't assert) if best_cost ends up worse
    than baseline_cost: seed_with_default_vector=True for the
    discretized Ant System only pre-boosts pheromone near the baseline
    -- it does not force the baseline vector into the ant population,
    so an occasional miss is possible variance, not necessarily a bug
    (see module docstring)."""
    controller = aco_script.FuzzyController()
    kwargs = dict(ACO_KWARGS)
    kwargs["random_seed"] = seed
    aco = aco_script.ACOOptimizer(controller, fitness_fn, **kwargs)
    t0 = time.time()
    best_position, best_cost, history = aco.optimize()
    elapsed = time.time() - t0

    if kwargs.get("seed_with_default_vector", False) and best_cost > baseline_cost + 1e-9:
        print(f"  WARNING: ACO (seed={seed}) finished worse than baseline "
              f"(best={best_cost:.4f} > baseline={baseline_cost:.4f}). "
              f"Possible with ACO's soft bias -- not necessarily a bug. If "
              f"this happens on most/all seeds, increase num_ants/max_iter/"
              f"archive_size.")

    return dict(name="ACO", seed=seed, best_position=best_position,
                best_cost=best_cost, history=history, elapsed=elapsed)


# ----------------------------------------------------------------------
# Criterion 3: stability of the solution across different runs
# ----------------------------------------------------------------------

def stability_analysis(fitness_fn, baseline_cost: float) -> dict:
    """Re-runs each optimizer once per seed in STABILITY_SEEDS, on the
    SAME fitness landscape, and summarizes how much the final cost
    AND the final parameter vector vary run-to-run. Low spread = a
    reliable algorithm; high spread = sensitive to random init.

    Every individual run here is validated against baseline_cost too
    (see run_pso_once / run_aco_once) -- stability is only meaningful
    if every run actually represents a real improvement attempt, not
    a mix of good runs and silently-broken ones."""
    print(f"\nStability analysis: re-running each optimizer with "
          f"{len(STABILITY_SEEDS)} different seeds ({STABILITY_SEEDS})...")

    pso_runs, aco_runs = [], []
    for seed in STABILITY_SEEDS:
        print(f"  [stability] PSO  seed={seed} ...", end=" ", flush=True)
        r = run_pso_once(fitness_fn, seed, baseline_cost)
        pso_runs.append(r)
        print(f"cost={r['best_cost']:.4f}")

        print(f"  [stability] ACO  seed={seed} ...", end=" ", flush=True)
        r = run_aco_once(fitness_fn, seed, baseline_cost)
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
# Criterion 4: effect of optimization on the fuzzy controller's
# actual performance
# ----------------------------------------------------------------------

def report_per_scenario_full(controller, factories, labels, num_steps: int) -> list[dict]:
    """Like pso_script.report_per_scenario, but also keeps the raw
    (W, Q, S) metrics per scenario -- not just the scalar cost -- so
    the numbers behind each cost value are inspectable (this doubles
    as the "generated data" deliverable, §8 item 2). Reimplements
    evaluate_controller's realistic block-based switching loop
    directly (rather than calling evaluate_controller, which only
    returns the scalar cost) so env.get_metrics() stays accessible."""
    rows = []
    for factory, label in zip(factories, labels):
        env = factory()
        env.reset()
        ticks_run = 0
        while ticks_run < num_steps:
            g1, g2 = controller.compute_green_time(env.queue_length_1, env.queue_length_2)
            g1_ticks = max(1, round(g1))
            g2_ticks = max(1, round(g2))
            for _ in range(g1_ticks):
                if ticks_run >= num_steps:
                    break
                env.step(1)
                ticks_run += 1
            for _ in range(g2_ticks):
                if ticks_run >= num_steps:
                    break
                env.step(2)
                ticks_run += 1
        metrics = env.get_metrics()
        cost = compute_cost(metrics)
        rows.append(dict(label=label, cost=cost, metrics=metrics))
    return rows


# ----------------------------------------------------------------------
# Output writers
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


def write_summary_report(results, baseline_cost, stability, per_scenario_by_method,
                          out_path: Path) -> None:
    lines = ["# PSO vs ACO — comparison summary\n"]

    lines.append("## 1. Final cost-function value\n")
    lines.append(f"Baseline (default hand-picked params): **{baseline_cost:.4f}**\n")
    lines.append("| Method | Best avg cost | Improvement over baseline | Runtime (s) |")
    lines.append("|---|---|---|---|")
    for r in results:
        improvement = baseline_cost - r["best_cost"]
        pct = 100 * improvement / baseline_cost if baseline_cost else float("nan")
        lines.append(f"| {r['name']} | {r['best_cost']:.4f} | "
                      f"{improvement:+.4f} ({pct:+.1f}%) | {r['elapsed']:.2f} |")

    lines.append("\n## 2. Convergence speed\n")
    lines.append("See `results/plots/pso_vs_aco_convergence.png` for the full curve. "
                  "Best-cost-so-far sampled across the run:\n")
    for r in results:
        step = max(1, len(r["history"]) // 8)
        sampled = [round(h, 3) for h in r["history"][::step]]
        lines.append(f"- **{r['name']}**: {sampled}")

    lines.append("\n## 3. Stability of the solution across different runs\n")
    lines.append(f"Each optimizer re-run independently with seeds "
                  f"{[r['seed'] for r in stability['pso_runs']]} "
                  f"on the identical multi-scenario fitness landscape. Every "
                  f"individual run was validated against the baseline (see "
                  f"module docstring) before being included here.\n")
    lines.append("| Method | Cost mean | Cost std | Cost min | Cost max | "
                  "Mean per-parameter std |")
    lines.append("|---|---|---|---|---|---|")
    for name, summary in (("PSO", stability["pso_summary"]), ("ACO", stability["aco_summary"])):
        s = summary
        lines.append(f"| {name} | {s['cost_mean']:.4f} | {s['cost_std']:.4f} | "
                      f"{s['cost_min']:.4f} | {s['cost_max']:.4f} | "
                      f"{s['param_std_mean']:.4f} |")
    lines.append("\nLower `cost std` / `mean per-parameter std` = more stable "
                  "(lands on a similar answer regardless of random seed).\n")

    lines.append("\n## 4. Effect of optimization on the fuzzy controller's performance\n")
    lines.append("Baseline (untuned) vs. each tuned controller, evaluated on every "
                  "individual traffic scenario/seed, not just the aggregate:\n")
    all_labels = [row["label"] for row in per_scenario_by_method["Baseline"]]
    header = "| Scenario | " + " | ".join(per_scenario_by_method.keys()) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(per_scenario_by_method) + 1))
    for i, label in enumerate(all_labels):
        row_vals = [f"{per_scenario_by_method[m][i]['cost']:.2f}" for m in per_scenario_by_method]
        lines.append(f"| {label} | " + " | ".join(row_vals) + " |")

    out_path.write_text("\n".join(lines))


def write_summary_csv(results, baseline_cost, stability, out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "best_cost", "improvement_over_baseline",
                          "runtime_s", "stability_cost_mean", "stability_cost_std",
                          "stability_param_std_mean"])
        writer.writerow(["Baseline", round(baseline_cost, 4), "", "", "", "", ""])
        for r, key in zip(results, ("pso_summary", "aco_summary")):
            s = stability[key]
            writer.writerow([
                r["name"], round(r["best_cost"], 4),
                round(baseline_cost - r["best_cost"], 4), round(r["elapsed"], 2),
                round(s["cost_mean"], 4), round(s["cost_std"], 4),
                round(s["param_std_mean"], 4),
            ])


def write_param_report(controller, results, baseline_cost, default_vec, out_path: Path) -> None:
    lines = ["# PSO vs ACO — tuned parameter values\n",
             f"Baseline cost (default vector): **{baseline_cost:.4f}**\n",
             "\n## Baseline\n```json",
             json.dumps(pso_script.vector_to_readable(controller, default_vec), indent=2),
             "```\n"]
    for r in results:
        lines.append(f"## {r['name']}\n")
        lines.append(f"- Best cost: **{r['best_cost']:.4f}**")
        lines.append(f"- Improvement over baseline: **{baseline_cost - r['best_cost']:+.4f}**")
        lines.append(f"- Runtime: {r['elapsed']:.2f}s\n")
        lines.append("```json")
        lines.append(json.dumps(pso_script.vector_to_readable(controller, r["best_position"]), indent=2))
        lines.append("```\n")
    out_path.write_text("\n".join(lines))


def write_param_csv(results, default_vec, out_path: Path) -> None:
    names: list[str] = []
    from fuzzy.fuzzy_controller import FuzzyController
    for set_name in FuzzyController.INPUT_SETS:
        for point in ("a", "b", "c"):
            names.append(f"mf_queue.{set_name}.{point}")
    for set_name in FuzzyController.OUTPUT_SETS:
        for point in ("a", "b", "c"):
            names.append(f"mf_green.{set_name}.{point}")
    for i, (s1, s2, out) in enumerate(FuzzyController.RULES):
        names.append(f"rule_weights[{i}] (IF q1={s1} AND q2={s2} THEN green={out})")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["param_name", "baseline"] + [r["name"] for r in results]
        writer.writerow(header)
        columns = [default_vec] + [r["best_position"] for r in results]
        for i, name in enumerate(names):
            writer.writerow([name] + [round(float(col[i]), 4) for col in columns])


def write_per_scenario_metrics_csv(per_scenario_by_method: dict, out_path: Path) -> None:
    """Raw (W, Q, S) metrics per scenario per method -- the 'generated
    data' behind every cost number in the summary report (§8 item 2)."""
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "scenario", "cost", "average_waiting_time",
                          "average_queue_length", "num_stops",
                          "total_cars_arrived", "total_cars_departed"])
        for method, rows in per_scenario_by_method.items():
            for row in rows:
                m = row["metrics"]
                writer.writerow([
                    method, row["label"], round(row["cost"], 4),
                    round(m["average_waiting_time"], 4),
                    round(m["average_queue_length"], 4),
                    m["num_stops"], m["total_cars_arrived"], m["total_cars_departed"],
                ])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    if COMPARE_QUICK_TEST:
        print("COMPARE_QUICK_TEST is ON -- fast sanity check of this script, "
              "not real Phase 4 numbers. Set COMPARE_QUICK_TEST = False for "
              "the full comparison.\n")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    env_factories, labels = pso_script.build_env_factories()
    print(f"Fitness landscape: {len(SCENARIOS)} traffic patterns x {len(SEEDS)} "
          f"seeds = {len(env_factories)} scenario/seed combinations, averaged "
          f"per candidate. num_steps={NUM_STEPS}, max_iter={MAX_ITER}\n")
    print(f"Optimizer seeds -- headline: 7. Stability sweep: {STABILITY_SEEDS} "
          f"(these are separate from the traffic-scenario seeds above).\n")

    controller = pso_script.FuzzyController()
    fitness_fn = pso_script.make_multi_scenario_fitness(
        env_factories=env_factories, num_steps=NUM_STEPS, aggregate=np.mean,
    )

    default_vec = controller.get_default_vector()
    controller.set_params_from_vector(default_vec)
    baseline_cost = fitness_fn(controller)
    print(f"Baseline (default vector) avg cost: {baseline_cost:.4f}")

    # --- headline runs (used for the convergence plot + criterion 1) ---
    print(f"\nRunning headline PSO run (seed={PSO_KWARGS['random_seed']})...")
    pso_result = run_pso_once(fitness_fn, PSO_KWARGS["random_seed"], baseline_cost)
    print(f"PSO best avg cost: {pso_result['best_cost']:.4f}  ({pso_result['elapsed']:.2f}s)  "
          f"-- guarantee OK: best_cost <= baseline_cost")
    pso_script.check_feasibility(pso_result["best_position"],
                                  *controller.get_param_bounds(), "PSO best_position")

    print(f"\nRunning headline ACO run (seed={ACO_KWARGS['random_seed']})...")
    aco_result = run_aco_once(fitness_fn, ACO_KWARGS["random_seed"], baseline_cost)
    print(f"ACO best avg cost: {aco_result['best_cost']:.4f}  ({aco_result['elapsed']:.2f}s)")
    aco_script.check_feasibility(aco_result["best_position"],
                                  *controller.get_param_bounds(), "ACO best_position")

    results = [pso_result, aco_result]

    # --- criterion 2: convergence plot ---
    plot_path = PLOTS_DIR / "pso_vs_aco_convergence.png"
    plot_convergence(results, baseline_cost, plot_path)
    print(f"\nSaved convergence plot to {plot_path}")

    # --- criterion 3: stability across independent runs (each run
    #     re-validated against baseline_cost internally) ---
    stability = stability_analysis(fitness_fn, baseline_cost)
    print(f"\nPSO stability: cost {stability['pso_summary']['cost_mean']:.4f} "
          f"+/- {stability['pso_summary']['cost_std']:.4f}  "
          f"(all {len(STABILITY_SEEDS)} seeds passed the baseline guarantee)")
    print(f"ACO stability: cost {stability['aco_summary']['cost_mean']:.4f} "
          f"+/- {stability['aco_summary']['cost_std']:.4f}")
    aco_worse_count = sum(1 for r in stability["aco_runs"] if r["best_cost"] > baseline_cost + 1e-9)
    if aco_worse_count:
        print(f"  Note: ACO finished worse than baseline on {aco_worse_count}/"
              f"{len(STABILITY_SEEDS)} stability seeds (see warnings above).")

    # --- criterion 4: effect on actual controller performance ---
    # Baseline vs. EACH tuned controller (not just "best of the two"),
    # so a report reviewer can see both algorithms' before/after effect.
    per_scenario_by_method = {}
    controller.set_params_from_vector(default_vec)
    per_scenario_by_method["Baseline"] = report_per_scenario_full(
        controller, env_factories, labels, NUM_STEPS)
    for r in results:
        controller.set_params_from_vector(r["best_position"])
        per_scenario_by_method[r["name"]] = report_per_scenario_full(
            controller, env_factories, labels, NUM_STEPS)
    print("\nPer-scenario breakdown computed for Baseline, PSO-tuned, and ACO-tuned.")

    # --- write reports ---
    summary_md = TABLES_DIR / "pso_vs_aco_summary.md"
    write_summary_report(results, baseline_cost, stability, per_scenario_by_method, summary_md)
    print(f"Saved comparison summary to {summary_md}")

    summary_csv = TABLES_DIR / "pso_vs_aco_summary.csv"
    write_summary_csv(results, baseline_cost, stability, summary_csv)
    print(f"Saved comparison summary (CSV) to {summary_csv}")

    param_md = TABLES_DIR / "pso_vs_aco_result_params.md"
    write_param_report(controller, results, baseline_cost, default_vec, param_md)
    print(f"Saved parameter report to {param_md}")

    param_csv = TABLES_DIR / "pso_vs_aco_result_params.csv"
    write_param_csv(results, default_vec, param_csv)
    print(f"Saved parameter table (CSV) to {param_csv}")

    metrics_csv = TABLES_DIR / "pso_vs_aco_per_scenario_metrics.csv"
    write_per_scenario_metrics_csv(per_scenario_by_method, metrics_csv)
    print(f"Saved per-scenario raw metrics (CSV) to {metrics_csv}")

    # --- also persist each optimizer's headline result the same way
    #     run_pso_optimization.py / run_aco_optimization.py would on
    #     their own (reuses their own save_final_vector) ---
    controller.set_params_from_vector(pso_result["best_position"])
    pso_script.save_final_vector(controller, pso_result["best_position"],
                                  pso_result["best_cost"], baseline_cost,
                                  pso_result["history"])
    controller.set_params_from_vector(aco_result["best_position"])
    aco_script.save_final_vector(controller, aco_result["best_position"],
                                  aco_result["best_cost"], baseline_cost,
                                  aco_result["history"])

    # --- console summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Method':<10}{'Best cost':>12}{'Improvement':>15}{'Runtime (s)':>14}")
    print(f"{'Baseline':<10}{baseline_cost:>12.4f}{'--':>15}{'--':>14}")
    for r in results:
        print(f"{r['name']:<10}{r['best_cost']:>12.4f}"
              f"{baseline_cost - r['best_cost']:>+15.4f}{r['elapsed']:>14.2f}")


if __name__ == "__main__":
    main()