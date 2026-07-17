"""
optimization/pso.py

Phase 3: Particle Swarm Optimization for tuning the FuzzyController's
27-parameter vector (9 queue-MF params + 9 green-time-MF params +
9 rule weights).

This module treats both the fuzzy controller and the traffic
simulation as black boxes. It only ever talks to them through the
public contract described in fuzzy_controller.py:

    controller.get_param_bounds()      -> (lower, upper) numpy arrays
    controller.get_default_vector()    -> starting-point numpy array
    controller.set_params_from_vector(x)
    evaluate_controller(controller, env_factory, num_steps, ...) -> float cost

No fuzzy-logic or simulation code lives here, and none of it is
modified by this module.

------------------------------------------------------------------
Vector layout (informs ONLY the repair step, nothing else in PSO
needs to know this -- see FuzzyController.params_to_vector):

    [0:3]   mf_queue "Low"    (a, b, c)
    [3:6]   mf_queue "Medium" (a, b, c)
    [6:9]   mf_queue "High"   (a, b, c)
    [9:12]  mf_green "Short"  (a, b, c)
    [12:15] mf_green "Medium" (a, b, c)
    [15:18] mf_green "Long"   (a, b, c)
    [18:27] rule_weights[0..8]

Within every (a, b, c) triplet the lower/upper bounds returned by
get_param_bounds() are identical across a, b, and c (e.g. all three
share [0, q_max] for a queue MF). That means clip-then-sort is a
safe repair: sorting only reorders values that were already within
the shared bounds, so the repaired triangle is still feasible and
still satisfies a <= b <= c.
------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from cost_function import evaluate_controller

# Number of (a, b, c) membership-function triangles at the front of
# the vector (6 triangles: 3 queue sets + 3 green-time sets).
# Rule weights (indices 18:27) need only a [0, 1] clip, no sorting.
_NUM_TRIANGLES = 6
_TRIANGLE_WIDTH = 3
_TRIANGLE_BLOCK_END = _NUM_TRIANGLES * _TRIANGLE_WIDTH  # 18

# A fitness function takes a *configured* controller (position already
# applied via set_params_from_vector) and returns a scalar cost to
# MINIMIZE. Keeping this as a swappable callable is what lets us
# upgrade from "single scenario" to "average over several scenarios"
# later without touching PSOOptimizer at all.
FitnessFn = Callable[[object], float]


def make_single_scenario_fitness(
    env_factory: Callable[[], object],
    num_steps: int,
    **cost_kwargs,
) -> FitnessFn:
    """
    Build a fitness function for the simple, single-scenario case
    described in the README: one env_factory, evaluated once per
    candidate.

    cost_kwargs are forwarded to evaluate_controller (alpha, beta,
    gamma only -- see cost_function.py). evaluate_controller now
    always uses realistic block-based light switching internally
    (Phase 2 fix), so there is no decision_interval or similar
    per-tick knob to configure here anymore.
    """

    def fitness(controller) -> float:
        return evaluate_controller(controller, env_factory, num_steps, **cost_kwargs)

    return fitness


def make_multi_scenario_fitness(
    env_factories: Sequence[Callable[[], object]],
    num_steps: int,
    aggregate: Callable[[Sequence[float]], float] = np.mean,
    **cost_kwargs,
) -> FitnessFn:
    """
    Build a fitness function that averages (or otherwise aggregates)
    cost over several traffic scenarios. Drop-in replacement for
    make_single_scenario_fitness -- PSOOptimizer doesn't care which
    one it was given.
    """

    def fitness(controller) -> float:
        costs = [
            evaluate_controller(controller, env_factory, num_steps, **cost_kwargs)
            for env_factory in env_factories
        ]
        return float(aggregate(costs))

    return fitness


@dataclass
class Particle:
    """One candidate fuzzy controller (one point in 27-D parameter space)."""

    position: np.ndarray
    velocity: np.ndarray
    fitness: float = field(default=np.inf)
    personal_best_position: np.ndarray = field(default=None)
    personal_best_cost: float = field(default=np.inf)

    def __post_init__(self) -> None:
        if self.personal_best_position is None:
            self.personal_best_position = self.position.copy()


class PSOOptimizer:
    """
    Standard (global-best) Particle Swarm Optimization over the
    FuzzyController's flat parameter vector.

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
        pso = PSOOptimizer(controller, fitness_fn, num_particles=30, max_iter=100)
        best_position, best_cost, history = pso.optimize()
    """

    def __init__(
        self,
        controller,
        fitness_fn: FitnessFn,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_min: Optional[float] = None,
        v_max_fraction: float = 0.2,
        random_seed: Optional[int] = None,
        seed_with_default_vector: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        controller : FuzzyController
            Any controller instance -- used only to read
            get_param_bounds() / get_default_vector() and, during
            evaluation, to have set_params_from_vector() called on
            it. Its fuzzy logic is never touched by this class.
        fitness_fn : FitnessFn
            Callable(controller) -> float. See
            make_single_scenario_fitness / make_multi_scenario_fitness.
        num_particles, max_iter : int
            Swarm size and number of iterations.
        w, c1, c2 : float
            Inertia weight, cognitive coefficient, social coefficient.
        w_min : float or None
            If given, inertia decays linearly from `w` down to
            `w_min` over the run (a common PSO refinement). If None,
            `w` stays constant.
        v_max_fraction : float
            Caps per-dimension velocity magnitude at this fraction of
            that dimension's (upper - lower) range, to avoid particles
            routinely overshooting the search space in one step.
        random_seed : int or None
            Seed for the swarm's own RNG, for reproducible runs.
        seed_with_default_vector : bool
            If True, one particle is initialized exactly at
            controller.get_default_vector() (a known-good baseline)
            instead of a fully random position.
        """
        self.controller = controller
        self.fitness_fn = fitness_fn

        self.num_particles = num_particles
        self.max_iter = max_iter

        self.w_start = w
        self.w_min = w_min if w_min is not None else w
        self.c1 = c1
        self.c2 = c2

        self.lower_bounds, self.upper_bounds = controller.get_param_bounds()
        self.dim = self.lower_bounds.shape[0]

        self.v_max = v_max_fraction * (self.upper_bounds - self.lower_bounds)

        self.rng = np.random.default_rng(random_seed)
        self.seed_with_default_vector = seed_with_default_vector

        self.swarm: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_cost: float = np.inf

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_population(self) -> None:
        """Create the swarm: random positions within bounds, zero (or
        small random) initial velocities, evaluate initial fitness."""
        self.swarm = []

        for i in range(self.num_particles):
            if i == 0 and self.seed_with_default_vector:
                position = self.controller.get_default_vector().copy()
            else:
                position = self.rng.uniform(self.lower_bounds, self.upper_bounds)

            position = self._repair(position)

            velocity = self.rng.uniform(-self.v_max, self.v_max)

            particle = Particle(position=position, velocity=velocity)
            particle.fitness = self._evaluate(particle.position)
            particle.personal_best_position = particle.position.copy()
            particle.personal_best_cost = particle.fitness

            self.swarm.append(particle)
            self._update_global_best(particle)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, position: np.ndarray) -> float:
        """Apply a candidate vector to the controller and score it.
        This is intentionally the only place PSO touches the
        controller/simulation black box."""
        self.controller.set_params_from_vector(position)
        return self.fitness_fn(self.controller)

    # ------------------------------------------------------------------
    # Repair: bounds + valid-triangle constraint
    # ------------------------------------------------------------------

    def _repair(self, position: np.ndarray) -> np.ndarray:
        """
        Enforce feasibility of a raw candidate position:
          1. Clip every dimension to [lower_bounds, upper_bounds].
          2. For each of the 6 membership-function triangles, sort its
             3 values ascending so a <= b <= c holds. This is safe
             because a, b, c share identical bounds within each
             triangle (see module docstring), so sorting cannot push
             any value out of range.
          3. Rule weights (already clipped to [0, 1] in step 1) need
             no further repair.
        """
        repaired = np.clip(position, self.lower_bounds, self.upper_bounds)

        # a ≤ b ≤ c
        for start in range(0, _TRIANGLE_BLOCK_END, _TRIANGLE_WIDTH):
            end = start + _TRIANGLE_WIDTH
            repaired[start:end] = np.sort(repaired[start:end])

        return repaired

    # ------------------------------------------------------------------
    # PSO update rules
    # ------------------------------------------------------------------

    def _current_inertia(self, iteration: int) -> float:
        """Linear inertia decay from w_start to w_min over max_iter
        iterations (constant if w_min was not configured)."""
        if self.max_iter <= 1:
            return self.w_start
        progress = iteration / (self.max_iter - 1)
        return self.w_start + progress * (self.w_min - self.w_start)

    def _update_velocity(self, particle: Particle, inertia: float) -> None:
        r1 = self.rng.uniform(0.0, 1.0, size=self.dim)
        r2 = self.rng.uniform(0.0, 1.0, size=self.dim)

        cognitive = self.c1 * r1 * (particle.personal_best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)

        particle.velocity = inertia * particle.velocity + cognitive + social
        particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)

    def _update_position(self, particle: Particle) -> None:
        particle.position = self._repair(particle.position + particle.velocity)

    def _update_personal_best(self, particle: Particle) -> None:
        if particle.fitness < particle.personal_best_cost:
            particle.personal_best_cost = particle.fitness
            particle.personal_best_position = particle.position.copy()

    def _update_global_best(self, particle: Particle) -> None:
        if particle.personal_best_cost < self.global_best_cost:
            self.global_best_cost = particle.personal_best_cost
            self.global_best_position = particle.personal_best_position.copy()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(self) -> tuple[np.ndarray, float, List[float]]:
        """
        Run the full PSO optimization loop.

        Returns
        -------
        best_position : np.ndarray
            The best 27-parameter vector found (global best).
        best_cost : float
            Its cost (lower is better).
        history : list[float]
            Global-best cost after each iteration, for convergence
            plots.
        """
        self._initialize_population()
        history: List[float] = [self.global_best_cost]

        for iteration in range(self.max_iter):
            inertia = self._current_inertia(iteration)

            for particle in self.swarm:
                self._update_velocity(particle, inertia)
                self._update_position(particle)
                particle.fitness = self._evaluate(particle.position)

                self._update_personal_best(particle)
                self._update_global_best(particle)

            history.append(self.global_best_cost)

        return self.global_best_position, self.global_best_cost, history