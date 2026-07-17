# `compare_optimizers.py`

`experiments/compare_optimizers.py` is the Phase 4 deliverable script:
it runs both PSO and ACO on the fuzzy controller and produces the four
comparison criteria the phase requires, as plots + tables you can drop
straight into a report. It supersedes the earlier
`compare_algorithms.py` smoke-test script — same spirit, much more
thorough.

If you haven't read it yet, `OPTIMIZERS_README.md` covers how
`PSOOptimizer`/`ACOOptimizer` themselves work internally; this document
only covers what *this script* does with them.

---

## 1. What it produces

| # | Criterion | How | Where |
|---|---|---|---|
| 1 | **Final cost-function value** achieved by each algorithm | One "headline" run per optimizer (`--seed`) | `pso_vs_aco_summary.md` §1, `.csv` |
| 2 | **Convergence speed** | Best-cost-so-far per iteration, plotted for both + baseline | `results/plots/pso_vs_aco_convergence.png`, sampled table in `.md` §2 |
| 3 | **Stability across runs** | Re-run each optimizer independently over several seeds (`--stability-seeds`); report mean/std of both final cost and final parameter vector | `pso_vs_aco_summary.md` §3, `.csv` |
| 4 | **Effect on actual controller performance** | The better-performing tuned controller evaluated on *every individual* traffic scenario/seed it was optimized against, not just the aggregate number the optimizer minimized | `pso_vs_aco_summary.md` §4 |

Plus the tuned parameter vectors themselves, human-readable:

```
results/tables/pso_vs_aco_result_params.md   (baseline + PSO + ACO, grouped by MF/rule name)
results/tables/pso_vs_aco_result_params.csv  (same, one row per parameter, machine-readable)
```

All five output files land under `results/plots/` and `results/tables/`
(or `<output-dir>/plots/` and `<output-dir>/tables/` if you pass
`--output-dir`).

---

## 2. Run it

```bash
# Default: full Phase 4 run (100 iterations, 5 stability seeds)
python experiments/compare_optimizers.py

# Common overrides
python experiments/compare_optimizers.py --max-iter 100 --stability-seeds 1 2 3 4 5

# Fast sanity check before committing to the full run
python experiments/compare_optimizers.py --max-iter 5 --skip-stability
```

Run it from the project root — it appends `src/` to `sys.path` itself,
so imports of `fuzzy.fuzzy_controller`, `cost_function`, and
`optimization.*` resolve without needing `PYTHONPATH` set manually.

### Full CLI reference

**Simulation / fitness landscape**

| Flag | Default | Meaning |
|---|---|---|
| `--num-steps` | `1000` | Simulation length per evaluation. |
| `--departure-rate` | `1.0` | Passed through to every `TrafficEnv`. |
| `--scenario-seeds` | `1 2 3` | Seeds crossed with the 4 fixed traffic patterns (moderate/heavy × symmetric/asymmetric — see below) to build the fitness landscape both optimizers are tuned against. `4 patterns × N seeds` scenario/seed combinations, averaged per candidate via `make_multi_scenario_fitness`. |

**Shared optimizer settings**

| Flag | Default | Meaning |
|---|---|---|
| `--max-iter` | `100` | Iterations for every PSO/ACO run in this script (headline and stability). |
| `--seed` | `7` | RNG seed for the **headline** run — the one used for the convergence plot and the per-scenario report (criteria 2 and 4). |
| `--stability-seeds` | `1 2 3 4 5` | Independent optimizer seeds for the stability analysis (criterion 3). Each seed triggers one full PSO run and one full ACO run. |

**PSO-specific** (forwarded straight to `PSOOptimizer`)

| Flag | Default |
|---|---|
| `--pso-particles` | `30` |
| `--pso-w` | `0.7` |
| `--pso-w-min` | `0.4` |
| `--pso-c1` | `1.5` |
| `--pso-c2` | `1.5` |

**ACO-specific** (forwarded straight to `ACOOptimizer` — note `q`/`xi`
are the re-purposed Ant-System deposit constant / evaporation rate, not
the old ACO_R parameters; see `OPTIMIZERS_README.md` §3 if that's
unfamiliar)

| Flag | Default |
|---|---|
| `--aco-archive-size` | `20` |
| `--aco-num-ants` | `10` |
| `--aco-q` | `0.3` |
| `--aco-xi` | `0.85` |

**Misc**

| Flag | Default | Meaning |
|---|---|---|
| `--skip-stability` | off | Skip criterion 3's re-runs entirely (they're the slowest part — `2 × len(stability_seeds)` full optimizer runs). Useful for a quick end-to-end check. When set, the summary report still gets a stability section, but it's just the headline run's own cost with zero spread — clearly not real stability data, just a placeholder so the report format doesn't break. |
| `--output-dir` | `results/` | Override where `plots/` and `tables/` subfolders are created. |

> **Note on `decision_interval`:** you won't find that flag here on
> purpose. `evaluate_controller()` no longer accepts it (removed as
> part of Phase 2's block-based light-switching fix) — the script's own
> comment flags this explicitly so nobody re-adds it and reintroduces
> the `TypeError` that broke the earlier `compare_algorithms.py`
> script.

### The four fixed traffic patterns

Hard-coded (not a CLI flag, so headline/stability/Phase-3 run scripts
all stay comparable):

```python
SCENARIOS = [
    ("moderate symmetric",  0.3, 0.3),
    ("moderate asymmetric", 0.4, 0.2),
    ("heavy asymmetric",    0.6, 0.2),
    ("heavy symmetric",     0.5, 0.5),
]
```

Each is crossed with every seed in `--scenario-seeds` to build the
full fitness landscape (`build_env_factories`). These are the same
patterns `run_pso_optimization.py`/`run_aco_optimization.py` use, so
"final cost" numbers from all these scripts are directly comparable to
each other.

---

## 3. What happens when you run it, in order

1. **Build the fitness landscape** — cross `SCENARIOS × --scenario-seeds` into a list of env factories, wrap them in `make_multi_scenario_fitness` (mean-aggregated).
2. **Baseline** — evaluate `FuzzyController().get_default_vector()` on that same fitness function, so every later "improvement over baseline" number is measured against something concrete.
3. **Headline runs** — one `PSOOptimizer` and one `ACOOptimizer`, both seeded with `--seed`, both run once via `.optimize()`. `check_feasibility()` asserts the returned vector is within bounds and every membership-function triangle is valid (`a <= b <= c`) — a hard failure here means something is broken upstream, not just a bad tuning result.
4. **Convergence plot** (criterion 2) — both headline `history` arrays plotted against iteration, with the baseline cost as a dashed reference line.
5. **Stability analysis** (criterion 3, unless `--skip-stability`) — re-runs *each* optimizer once per seed in `--stability-seeds` (so `2 × len(stability_seeds)` total optimizer runs — this is the expensive part). For each algorithm, summarizes:
   - `cost_mean`, `cost_std`, `cost_min`, `cost_max` across those runs' final costs
   - `param_std_mean` — per-dimension std of the final 27-vector across runs, averaged into one number, so you get a sense of parameter-level (not just cost-level) reproducibility too
6. **Per-scenario report** (criterion 4) — whichever of the two headline results has the lower final cost gets loaded into the controller, then evaluated individually on every one of the scenario/seed combinations from step 1, so a good average can't be hiding a regression on one specific traffic pattern.
7. **Write everything to disk** and print a console summary table.

---

## 4. Reading the output files

**`pso_vs_aco_summary.md`** is the main deliverable — four numbered
sections matching the four criteria above, written as Markdown tables.
Section 3 explicitly explains the direction of the stability numbers
("lower cost std / param std = more reliable, algorithm lands on a
similar answer regardless of seed").

**`pso_vs_aco_summary.csv`** is the same headline + stability numbers,
one row per method (`Baseline`, `PSO`, `ACO`), for pulling into a
spreadsheet or another plotting tool.

**`pso_vs_aco_result_params.md` / `.csv`** show the actual tuned
27-vectors — baseline vs. PSO vs. ACO, grouped by membership-function
name and rule description (`vector_to_readable`) in the `.md`, and as
one row per parameter name (`flat_param_names`, in the exact order
`FuzzyController.params_to_vector` produces) in the `.csv`.

**`pso_vs_aco_convergence.png`** — one figure, both algorithms' curves
plus the baseline as a dashed horizontal line, log-free linear axes
(so a controller that fails to improve at all is immediately obvious
as a flat line at/above the baseline).

---

## 5. Relationship to the other Phase 3/4 scripts

| Script | Purpose |
|---|---|
| `run_pso_optimization.py` | One deep PSO run only, with live per-iteration ETA logging and final-vector logging to `results/logs/`. Good for producing *the* tuned PSO controller you'll actually deploy. |
| `run_aco_optimization.py` | Same, for ACO only. |
| `compare_optimizers.py` (this script) | Runs both, adds stability analysis across multiple seeds, and produces the full four-criterion Phase 4 report in one pass. Not meant to replace the two scripts above for producing a final deployed controller — it's the *comparison* deliverable. |

All three use the same `SCENARIOS` list, the same
`make_multi_scenario_fitness` helper, and the same `PSOOptimizer`/
`ACOOptimizer` classes, so numbers are consistent across all of them —
none of them is evaluating on an easier or different landscape than the
others.

---

## 6. Tips

- **First run:** use `--max-iter 5 --skip-stability` to confirm the
  whole pipeline runs end-to-end (imports resolve, output dirs get
  created, feasibility checks pass) before committing to a full
  100-iteration × 5-seed run, which can take a while — see the
  per-iteration ETA logging in `run_pso_optimization.py` /
  `run_aco_optimization.py` for a sense of how long one optimizer run
  takes with your current `--num-steps`/`--max-iter`, then multiply
  by roughly `2 × (1 + len(--stability-seeds))` total optimizer runs
  for this script's full pipeline.
- **Changing `--scenario-seeds` or `--num-steps`** changes the fitness
  landscape itself, so costs from that run won't be numerically
  comparable to a run with different values — keep these fixed across
  a PSO-vs-ACO comparison you intend to report.
- **`--output-dir`** is handy for keeping multiple comparison runs
  (e.g. different hyperparameter settings) from overwriting each
  other's plots/tables.