"""
optimization/aco.py

Phase 3: Ant Colony Optimization for tuning the FuzzyController's
27-parameter vector, using DISCRETIZED "classic" Ant System ACO
(Dorigo, 1992) rather than ACO_R.

Why discretization is necessary
--------------------------------
Classic ACO is defined over discrete choices (edges in a graph):
a finite set of options at each step, a real pheromone value per
option, probabilistic selection weighted by pheromone, and
evaporation + deposit as the update rule (Delta_tau_k = Q / cost_k,
deposited on the edges ant k used -- exactly Dorigo's original Ant
System formula).

Our search space is a continuous 27-dimensional real vector, so
there is no native graph to run that algorithm on. This module
manufactures one: each of the 27 dimensions is sliced into
`num_bins` evenly-spaced discrete levels between its lower and upper
bound, so "build a solution" becomes "for each dimension, pick one
of num_bins levels" -- a genuine discrete choice, with a genuine
pheromone table tau[dim, bin] and genuine evaporation/deposit. This
is the standard way "real" ACO is forced onto continuous domains
(as opposed to ACO_R, which sidesteps discretization entirely with
a Gaussian-kernel archive -- see git history for that version).

Trade-off vs ACO_R: parameters snap to bin centers instead of
varying continuously, so resolution is capped by `num_bins`. For a
smooth numeric landscape like this one, ACO_R is expected to
out-perform this version -- that's *why* ACO_R exists -- but this
is the "textbook" pheromone-trail algorithm, faithfully applied.

Like pso.py, this module treats the fuzzy controller and simulation
as black boxes and only calls:

    controller.get_param_bounds()
    controller.get_default_vector()
    controller.set_params_from_vector(x)
    evaluate_controller(...)   (indirectly, via a FitnessFn)

------------------------------------------------------------------
Vector layout (only relevant to the repair step -- see
FuzzyController.params_to_vector for the authoritative version):

    [0:3]   mf_queue "Low"    (a, b, c)
    [3:6]   mf_queue "Medium" (a, b, c)
    [6:9]   mf_queue "High"   (a, b, c)
    [9:12]  mf_green "Short"  (a, b, c)
    [12:15] mf_green "Medium" (a, b, c)
    [15:18] mf_green "Long"   (a, b, c)
    [18:27] rule_weights[0..8]

As before: within each (a, b, c) triangle the lower/upper bounds are
identical across a, b, and c, so all three dimensions of a triangle
share the same bin grid. That means clip-then-sort is still a safe,
bound-preserving repair after bins are decoded to values -- sorting
only reorders values drawn from the same shared grid.
------------------------------------------------------------------

Parameter-name mapping from the old ACO_R constructor
-------------------------------------------------------
compare_algorithms.py (and anything else already wired up) calls
this class with archive_size / num_ants / max_iter / q / xi /
random_seed. To stay a drop-in replacement, those names are kept
and re-purposed as follows:

    archive_size -> size of the initial random-sampling batch used
                     to "prime" the pheromone table before the main
                     loop starts (ACO_R used this as archive K; here
                     there is no archive, so it becomes an init
                     batch size instead).
    num_ants     -> ants constructed per iteration (same meaning as
                     before).
    max_iter     -> number of iterations (same meaning as before).
    q            -> reused as the Ant-System deposit constant Q in
                     Delta_tau = Q / cost (was ACO_R's locality
                     parameter -- different role, same name).
    xi           -> reused as the pheromone evaporation rate rho,
                     clipped to (0.01, 0.99) (was ACO_R's spread
                     decay parameter -- different role, same name).

New, ACO-specific knobs (num_bins, alpha, tau0, tau_min,
elitist_weight) are exposed as extra keyword arguments with sensible
defaults, so existing call sites don't need to change at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Reuse the fitness-function contract and helpers from pso.py rather
# than redefining them -- both algorithms optimize the exact same
# black-box objective, so there is only one definition of "how do we
# score a controller" for the whole project.
from optimization.pso import FitnessFn, make_single_scenario_fitness, make_multi_scenario_fitness  # noqa: F401

_TRIANGLE_BLOCK_END = 18
_TRIANGLE_WIDTH = 3

# Numerical floor to avoid dividing by zero when a candidate scores
# a cost of exactly 0.0 (deposit would otherwise be infinite).
_COST_EPS = 1e-9


@dataclass
class Ant:
    """One constructed candidate solution, plus the bin index it used
    in every dimension (needed so pheromone deposit knows which table
    cells to reinforce)."""

    position: np.ndarray
    bin_indices: np.ndarray  # shape (dim,), dtype int
    fitness: float = np.inf


class ACOOptimizer:
    """
    Discretized Ant System: classic pheromone-trail ACO applied to a
    binned version of the FuzzyController's flat parameter vector.

    Usage
    -----
        controller = FuzzyController()
        fitness_fn = make_single_scenario_fitness(
            env_factory=lambda: TrafficEnv(arrival_rate_1=0.4,
                                            arrival_rate_2=0.2,
                                            departure_rate=1.0,
                                            seed=42),
            num_steps=500,
        )
        aco = ACOOptimizer(controller, fitness_fn, archive_size=20, num_ants=10, max_iter=100)
        best_position, best_cost, history = aco.optimize()
    """

    def __init__(
        self,
        controller,
        fitness_fn: FitnessFn,
        archive_size: int = 20,
        num_ants: int = 10,
        max_iter: int = 100,
        q: float = 0.3,
        xi: float = 0.85,
        random_seed: Optional[int] = None,
        seed_with_default_vector: bool = False,
        num_bins: int = 20,
        alpha: float = 1.0,
        tau0: float = 1.0,
        tau_min: float = 1e-3,
        elitist_weight: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        controller : FuzzyController
            Used only to read get_param_bounds() / get_default_vector()
            and, during evaluation, to have set_params_from_vector()
            called on it. Its fuzzy logic is never touched here.
        fitness_fn : FitnessFn
            Callable(controller) -> float to MINIMIZE. See
            make_single_scenario_fitness / make_multi_scenario_fitness
            in pso.py.
        archive_size : int
            Size of the initial random-sampling batch used to prime
            the pheromone table before the main loop (see module
            docstring's parameter-mapping note).
        num_ants : int
            Number of new solutions constructed per iteration.
        max_iter : int
            Number of iterations (construct -> evaluate -> update
            pheromone cycles).
        q : float
            Ant-System deposit constant Q, used as
            Delta_tau = Q / cost for every ant (see module docstring's
            parameter-mapping note).
        xi : float
            Reused as the pheromone evaporation rate rho in
            [0.01, 0.99] (see module docstring's parameter-mapping
            note). Higher = faster forgetting of old pheromone,
            faster convergence, more risk of premature convergence.
        random_seed : int or None
            Seed for the colony's own RNG, for reproducible runs.
        seed_with_default_vector : bool
            If True, the pheromone table is pre-boosted at the bin
            nearest to controller.get_default_vector() in every
            dimension, biasing (not fixing) early exploration toward
            the hand-tuned baseline.
        num_bins : int
            Number of discrete levels each of the 27 dimensions is
            sliced into. More bins = finer resolution but a bigger
            table to learn (slower convergence for the same ant
            budget).
        alpha : float
            Pheromone-influence exponent: selection probability for a
            bin is proportional to tau ** alpha. alpha=1.0 is
            standard Ant System; higher values make the colony more
            greedily follow the strongest trails.
        tau0 : float
            Initial (and evaporation floor baseline) pheromone value
            in every cell.
        tau_min : float
            Hard floor on pheromone after evaporation, so no bin ever
            reaches exactly zero probability (keeps exploration
            alive, as in Max-Min Ant System).
        elitist_weight : float
            Extra deposit multiplier applied to the global-best
            solution's bins on every update, on top of its own
            regular Q/cost deposit (elitist reinforcement, a common
            Ant System refinement).
        """
        self.controller = controller
        self.fitness_fn = fitness_fn

        self.init_batch = archive_size
        self.num_ants = num_ants
        self.max_iter = max_iter

        # Re-purposed names -- see module docstring.
        self.Q = max(q, 1e-6)
        self.rho = float(np.clip(xi, 0.01, 0.99))

        self.num_bins = num_bins
        self.alpha = alpha
        self.tau0 = tau0
        self.tau_min = tau_min
        self.elitist_weight = elitist_weight

        self.lower_bounds, self.upper_bounds = controller.get_param_bounds()
        self.dim = self.lower_bounds.shape[0]

        self.rng = np.random.default_rng(random_seed)
        self.seed_with_default_vector = seed_with_default_vector

        # Bin centers per dimension: bin_centers[d, b] is the decoded
        # value used whenever dimension d's bin b is chosen.
        span = (self.upper_bounds - self.lower_bounds)[:, None]
        offsets = (np.arange(num_bins) + 0.5) / num_bins  # shape (num_bins,)
        self.bin_centers = self.lower_bounds[:, None] + offsets[None, :] * span  # (dim, num_bins)

        # The pheromone table itself -- this IS the "memory" that
        # replaces ACO_R's archive. One real number per (dimension,
        # bin) cell.
        self.pheromone = np.full((self.dim, self.num_bins), self.tau0)

        if self.seed_with_default_vector:
            self._bias_toward_default_vector()

        self.best_position: Optional[np.ndarray] = None
        self.best_bin_indices: Optional[np.ndarray] = None
        self.best_cost: float = np.inf

    # ------------------------------------------------------------------
    # Initial bias (optional)
    # ------------------------------------------------------------------

    def _bias_toward_default_vector(self, boost_factor: float = 5.0) -> None:
        """Pre-boost the pheromone at the bin nearest to each dimension
        of the hand-tuned default vector. This is a soft bias, not a
        guarantee the default value is ever sampled -- consistent with
        the discrete, probabilistic nature of ACO selection."""
        default_vector = self.controller.get_default_vector()
        for d in range(self.dim):
            nearest_bin = int(np.argmin(np.abs(self.bin_centers[d] - default_vector[d])))
            self.pheromone[d, nearest_bin] *= boost_factor

    # ------------------------------------------------------------------
    # Repair: bounds + valid-triangle constraint (identical strategy to pso.py)
    # ------------------------------------------------------------------

    def _repair(self, position: np.ndarray) -> np.ndarray:
        """Clip to bounds (a no-op in practice since bin centers are
        already within bounds -- kept for safety/symmetry with
        pso.py), then sort each (a, b, c) triangle ascending so
        a <= b <= c holds. Safe because a, b, c share an identical bin
        grid within each triangle (see module docstring)."""
        repaired = np.clip(position, self.lower_bounds, self.upper_bounds)
        for start in range(0, _TRIANGLE_BLOCK_END, _TRIANGLE_WIDTH):
            end = start + _TRIANGLE_WIDTH
            repaired[start:end] = np.sort(repaired[start:end])
        return repaired

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, position: np.ndarray) -> float:
        """Apply a candidate vector to the controller and score it.
        This is intentionally the only place ACO touches the
        controller/simulation black box."""
        self.controller.set_params_from_vector(position)
        return self.fitness_fn(self.controller)

    # ------------------------------------------------------------------
    # Solution construction (the "ant walk" -- now a genuine discrete
    # walk over the pheromone table, one bin choice per dimension)
    # ------------------------------------------------------------------

    def _construct_ant(self) -> Ant:
        """
        Build one new candidate solution, one dimension at a time: for
        each dimension, choose a bin via roulette-wheel selection
        weighted by tau[dim, :] ** alpha (Dorigo's classic transition
        rule, with no heuristic/visibility term since there is no
        natural "distance" between bins here -- pheromone-only
        selection). Decode the chosen bins to a real-valued vector via
        bin_centers, then repair.
        """
        bin_indices = np.empty(self.dim, dtype=int)
        raw_position = np.empty(self.dim)

        for dim_idx in range(self.dim):
            weights = self.pheromone[dim_idx] ** self.alpha
            probs = weights / weights.sum()
            chosen_bin = self.rng.choice(self.num_bins, p=probs)

            bin_indices[dim_idx] = chosen_bin
            raw_position[dim_idx] = self.bin_centers[dim_idx, chosen_bin]

        position = self._repair(raw_position)
        fitness = self._evaluate(position)
        return Ant(position=position, bin_indices=bin_indices, fitness=fitness)

    # ------------------------------------------------------------------
    # Pheromone update: evaporation + deposit (the actual "ACO" part)
    # ------------------------------------------------------------------

    def _update_pheromone(self, ants: List[Ant]) -> None:
        """
        Classic Ant System pheromone update, applied once per batch
        (both the initial priming batch and every subsequent
        iteration):

          1. Evaporation: every cell decays by factor (1 - rho),
             floored at tau_min so no bin's probability collapses to
             exactly zero (Max-Min-style safeguard against premature
             stagnation).
          2. Deposit: each ant reinforces the bins it used by
             Delta_tau = Q / cost -- better (lower-cost) solutions
             deposit more pheromone, exactly Dorigo's original rule.
          3. Elitist deposit: the best solution found so far gets an
             extra reinforcement on top of its own regular deposit,
             on every update (a common, well-established Ant System
             refinement that keeps the best-known trail from being
             evaporated away).
        """
        self.pheromone *= (1.0 - self.rho)
        self.pheromone = np.maximum(self.pheromone, self.tau_min)

        for ant in ants:
            deposit = self.Q / (ant.fitness + _COST_EPS)
            self.pheromone[np.arange(self.dim), ant.bin_indices] += deposit

        if self.best_bin_indices is not None:
            elite_deposit = self.elitist_weight * self.Q / (self.best_cost + _COST_EPS)
            self.pheromone[np.arange(self.dim), self.best_bin_indices] += elite_deposit

    def _update_global_best(self, ants: List[Ant]) -> None:
        for ant in ants:
            if ant.fitness < self.best_cost:
                self.best_cost = ant.fitness
                self.best_position = ant.position.copy()
                self.best_bin_indices = ant.bin_indices.copy()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(self) -> tuple[np.ndarray, float, List[float]]:
        """
        Run the full discretized-ACO optimization loop.

        Returns
        -------
        best_position : np.ndarray
            The best 27-parameter vector found.
        best_cost : float
            Its cost (lower is better).
        history : list[float]
            Global-best cost after the initial priming batch and
            after each subsequent iteration, for convergence plots.
            Monotonically non-increasing by construction (global best
            is tracked separately from the pheromone table, so
            evaporation can never make it worse).
        """
        # Initial priming batch: with a freshly-uniform pheromone
        # table, selection probabilities are uniform too, so this is
        # equivalent to random sampling -- the discretized analogue of
        # ACO_R's random archive initialization.
        init_ants = [self._construct_ant() for _ in range(self.init_batch)]
        self._update_global_best(init_ants)
        self._update_pheromone(init_ants)

        history: List[float] = [self.best_cost]

        for _ in range(self.max_iter):
            ants = [self._construct_ant() for _ in range(self.num_ants)]
            self._update_global_best(ants)
            self._update_pheromone(ants)
            history.append(self.best_cost)

        return self.best_position, self.best_cost, history