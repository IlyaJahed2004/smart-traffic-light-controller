"""
optimization/aco.py

Phase 3: Ant Colony Optimization for tuning the FuzzyController's
27-parameter vector, using ACO_R (Ant Colony Optimization for
continuous domains -- Socha & Dorigo, 2008).

Why ACO_R and not "classic" graph/pheromone-trail ACO
------------------------------------------------------
Classic ACO is defined over discrete choices (edges in a graph).
Our search space is a continuous 27-dimensional real vector, so
there is no natural graph to lay pheromone on. ACO_R adapts the same
core idea -- solutions influence where future ants search, weighted
by quality -- to continuous variables:

  - Instead of a pheromone table, we keep an ARCHIVE of the best K
    solutions found so far, sorted by fitness (best first).
  - Each archive member acts as the center of a Gaussian "pheromone
    kernel". Better-ranked solutions get a higher probability of
    being chosen as a kernel (via Gaussian-weighted ranks).
  - Each new ant builds a solution dimension-by-dimension: for each
    dimension, pick a kernel (weighted by rank) and sample a Gaussian
    centered on that kernel's value, with a spread proportional to
    how spread out the archive already is in that dimension.
  - New ants are merged into the archive, and only the best K
    solutions survive (elitist truncation) -- this is the "pheromone
    update" step.

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

As in pso.py: within each (a, b, c) triangle the lower/upper bounds
are identical across a, b, and c, so clip-then-sort is a safe,
bound-preserving repair.
------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Reuse the fitness-function contract and helpers from pso.py rather
# than redefining them -- both algorithms optimize the exact same
# black-box objective, so there is only one definition of "how do we
# score a controller" for the whole project.
from optimization.pso import FitnessFn, make_single_scenario_fitness, make_multi_scenario_fitness  # noqa: F401

_TRIANGLE_BLOCK_END = 18
_TRIANGLE_WIDTH = 3


@dataclass
class Ant:
    """One constructed candidate solution."""

    position: np.ndarray
    fitness: float = np.inf


class ACOOptimizer:
    """
    ACO_R: Ant Colony Optimization for continuous domains, applied to
    the FuzzyController's flat parameter vector.

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
        seed_with_default_vector: bool = True,
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
            Number of solutions (K) kept in the archive -- the ACO_R
            analogue of PSO's swarm size / pheromone memory.
        num_ants : int
            Number of new solutions constructed per iteration.
        max_iter : int
            Number of iterations (construct -> evaluate -> merge ->
            truncate cycles).
        q : float
            Locality parameter. Small q concentrates selection
            probability on the best-ranked archive solutions
            (exploitation); larger q spreads it more evenly across the
            whole archive (exploration).
        xi : float
            Convergence-speed parameter. Larger xi shrinks the Gaussian
            sampling spread faster as the archive converges, similar in
            spirit to PSO's inertia decay.
        random_seed : int or None
            Seed for the colony's own RNG, for reproducible runs.
        seed_with_default_vector : bool
            If True, one archive slot is initialized exactly at
            controller.get_default_vector() instead of a fully random
            position.
        """
        if archive_size < 2:
            raise ValueError("archive_size must be >= 2 (need at least 2 solutions "
                              "to estimate per-dimension spread)")

        self.controller = controller
        self.fitness_fn = fitness_fn

        self.archive_size = archive_size
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.q = q
        self.xi = xi

        self.lower_bounds, self.upper_bounds = controller.get_param_bounds()
        self.dim = self.lower_bounds.shape[0]

        self.rng = np.random.default_rng(random_seed)
        self.seed_with_default_vector = seed_with_default_vector

        # Archive is kept as two parallel arrays (positions, fitness)
        # plus the fixed rank-based selection weights (constant for a
        # given archive_size and q, so computed once).
        self.archive_positions: Optional[np.ndarray] = None  # shape (archive_size, dim)
        self.archive_fitness: Optional[np.ndarray] = None    # shape (archive_size,)
        self.selection_weights = self._compute_selection_weights()
        self.selection_probs = self.selection_weights / self.selection_weights.sum()

    # ------------------------------------------------------------------
    # Repair: bounds + valid-triangle constraint (identical strategy to pso.py)
    # ------------------------------------------------------------------

    def _repair(self, position: np.ndarray) -> np.ndarray:
        """Clip to bounds, then sort each (a, b, c) triangle ascending
        so a <= b <= c holds. Safe because a, b, c share identical
        bounds within each triangle (see module docstring)."""
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
    # Archive (the ACO_R "pheromone" structure)
    # ------------------------------------------------------------------

    def _compute_selection_weights(self) -> np.ndarray:
        """
        Gaussian weight per archive rank (rank 1 = best). Lower q
        concentrates weight on top ranks; higher q flattens it. This
        matches the standard ACO_R weighting:

            w_l = 1 / (q * K * sqrt(2*pi)) * exp(-(l-1)^2 / (2*q^2*K^2))

        where l is the 1-indexed rank and K is the archive size.
        """
        ranks = np.arange(1, self.archive_size + 1)
        k = self.archive_size
        weights = (1.0 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(
            -((ranks - 1) ** 2) / (2 * (self.q ** 2) * (k ** 2))
        )
        return weights

    def _initialize_archive(self) -> None:
        """Build the initial archive: random feasible solutions
        (optionally seeding one with the default vector), evaluated
        and sorted best-first."""
        positions = np.empty((self.archive_size, self.dim))

        for i in range(self.archive_size):
            if i == 0 and self.seed_with_default_vector:
                positions[i] = self.controller.get_default_vector().copy()
            else:
                positions[i] = self.rng.uniform(self.lower_bounds, self.upper_bounds)
            positions[i] = self._repair(positions[i])

        fitness = np.array([self._evaluate(p) for p in positions])

        self.archive_positions, self.archive_fitness = self._sort_by_fitness(positions, fitness)

    @staticmethod
    def _sort_by_fitness(positions: np.ndarray, fitness: np.ndarray):
        """Sort (positions, fitness) ascending by fitness (best/lowest first)."""
        order = np.argsort(fitness)
        return positions[order], fitness[order]

    # ------------------------------------------------------------------
    # Solution construction (the "ant walk")
    # ------------------------------------------------------------------

    def _construct_ant(self) -> Ant:
        """
        Build one new candidate solution, one dimension at a time:
        for each dimension, pick a guiding archive member (weighted by
        rank) and sample a Gaussian centered on that member's value in
        that dimension, with spread set by how dispersed the archive
        already is along that dimension (scaled by xi).
        """
        new_position = np.empty(self.dim)
        k = self.archive_size

        for dim_idx in range(self.dim):
            guide_idx = self.rng.choice(k, p=self.selection_probs)
            mean = self.archive_positions[guide_idx, dim_idx]

            # Average distance from the guide to every other archive
            # member along this dimension -- ACO_R's stand-in for
            # "how much has the colony already agreed on this variable".
            spread = np.sum(np.abs(self.archive_positions[:, dim_idx] - mean)) / (k - 1)
            sigma = self.xi * spread

            if sigma <= 0.0:
                # Archive has fully converged in this dimension; sample
                # tightly around the guide instead of collapsing to a
                # zero-variance (i.e. always-identical) draw.
                sigma = 1e-6

            new_position[dim_idx] = self.rng.normal(mean, sigma)

        new_position = self._repair(new_position)
        fitness = self._evaluate(new_position)
        return Ant(position=new_position, fitness=fitness)

    def _update_archive(self, new_ants: List[Ant]) -> None:
        """Merge newly constructed ants into the archive and keep only
        the best `archive_size` solutions overall (elitist truncation
        -- this is ACO_R's pheromone-update step)."""
        combined_positions = np.vstack(
            [self.archive_positions] + [ant.position for ant in new_ants]
        )
        combined_fitness = np.concatenate(
            [self.archive_fitness] + [[ant.fitness] for ant in new_ants]
        )

        sorted_positions, sorted_fitness = self._sort_by_fitness(combined_positions, combined_fitness)
        self.archive_positions = sorted_positions[: self.archive_size]
        self.archive_fitness = sorted_fitness[: self.archive_size]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(self) -> tuple[np.ndarray, float, List[float]]:
        """
        Run the full ACO_R optimization loop.

        Returns
        -------
        best_position : np.ndarray
            The best 27-parameter vector found (archive's rank-1 member).
        best_cost : float
            Its cost (lower is better).
        history : list[float]
            Best archive cost after each iteration, for convergence
            plots. Monotonically non-increasing by construction
            (elitist truncation never lets the best solution get worse).
        """
        self._initialize_archive()
        history: List[float] = [float(self.archive_fitness[0])]

        for _ in range(self.max_iter):
            new_ants = [self._construct_ant() for _ in range(self.num_ants)]
            self._update_archive(new_ants)
            history.append(float(self.archive_fitness[0]))

        best_position = self.archive_positions[0].copy()
        best_cost = float(self.archive_fitness[0])
        return best_position, best_cost, history