"""
fuzzy_controller.py

Phase 2: Mamdani-type fuzzy controller that decides the green-light
duration for Road 1 based on the current queue lengths of both roads.
Road 2's green time is derived from Road 1's (see `cycle_time` below).

This module is designed to be driven by:
    - a fixed/default parameter set (baseline controller)
    - PSO or ACO (Phase 3), which search over the SAME parameter
      vector this module exposes via `get_default_params()`,
      `params_to_vector()`, and `vector_to_params()`.

============================================================
INTERFACE CONTRACT FOR PHASE 3 (PSO / ACO) -- READ THIS FIRST
============================================================
Per the project spec, PSO/ACO must tune:
    1. Membership function parameters (for inputs AND output)
    2. Fuzzy rule weights

Everything tunable is packed into ONE flat numpy vector so that:
    - a PSO "particle position" IS this vector
    - an ACO "ant's constructed solution" IS this vector

Usage from optimization code (pso.py / aco.py), without needing to
know anything about fuzzy logic internals:

    from fuzzy.fuzzy_controller import FuzzyController

    controller = FuzzyController()
    default_vec = controller.get_default_vector()   # starting point / bounds reference
    bounds = controller.get_param_bounds()           # list of (min, max) per dimension

    # --- inside PSO/ACO's cost evaluation for a candidate vector `x` ---
    controller.set_params_from_vector(x)
    green_1 = controller.compute_green_time(queue_1, queue_2)
    # ... run TrafficEnv using this controller, then read env.get_metrics()
    # and compute C = alpha*W + beta*Q + gamma*S

That's the entire contract. Nothing else in this file needs to be
understood to write PSO/ACO -- just: vector in, `compute_green_time`
out, bounds available for initialization/clamping.
"""

import numpy as np


class FuzzyController:
    """
    Mamdani fuzzy controller with:
      Inputs  : queue_length_1, queue_length_2   (fuzzy sets: Low, Medium, High)
      Output  : green_time_1                     (fuzzy sets: Short, Medium, Long)
      Defuzz  : centroid method

    Road 2's green time is computed as `cycle_time - green_time_1`
    (clipped to a minimum green time), since the spec defines a
    single fuzzy output (green time for road 1).
    """

    # Fuzzy set names, fixed by the project spec -- do not need tuning
    INPUT_SETS = ["Low", "Medium", "High"]
    OUTPUT_SETS = ["Short", "Medium", "Long"]

    # Rule base, fixed structure (3x3 = 9 rules covering all input
    # combinations). Only the CONSEQUENT and its WEIGHT are tunable
    # (weight tuning is part of the PSO/ACO parameter vector).
    # Each rule: (queue1_set, queue2_set) -> output_set
    RULES = [
        ("Low",    "Low",    "Medium"),
        ("Low",    "Medium", "Short"),
        ("Low",    "High",   "Short"),
        ("Medium", "Low",    "Long"),
        ("Medium", "Medium", "Medium"),
        ("Medium", "High",   "Short"),
        ("High",   "Low",    "Long"),
        ("High",   "Medium", "Long"),
        ("High",   "High",   "Medium"),
    ]

    def __init__(
        self,
        max_queue_for_scaling=20,
        min_green_time=5,
        max_green_time=45,
        cycle_time=None,
        defuzz_resolution=101,
        params=None,
    ):
        """
        Parameters
        ----------
        max_queue_for_scaling : int
            Queue length considered "definitely High" for input membership
            functions. Used only to set sensible default MF ranges.
        min_green_time, max_green_time : int
            Bounds (in seconds) for the output universe of discourse.
            Also define the valid range PSO/ACO can search within for
            the green-time membership function parameters (see
            `get_param_bounds`).
        cycle_time : int or None
            Fixed total green time budget per full cycle (road 1 + road 2):
            green_time_2 = cycle_time - green_time_1 (clipped to
            [min_green_time, max_green_time]).

            IMPORTANT: defaults to None, which means it is AUTOMATICALLY
            SET to `min_green_time + max_green_time`. Do not override this
            with an unrelated fixed number -- doing so can silently break
            the invariant `green_1 + green_2 == cycle_time`.

            Concrete example of the bug this avoids: if someone sets
            max_green_time=48 while leaving a hardcoded cycle_time=50,
            a legitimately-computed green_1=48 would force
            green_2 = cycle_time - green_1 = 2, which then gets clipped
            UP to min_green_time=5 (since 2 < 5), making the true sum
            48 + 5 = 53 instead of 50 -- silently, with no error. Deriving
            cycle_time from the bounds instead guarantees this can never
            happen: if green_1 is within [min_green_time, max_green_time],
            then cycle_time - green_1 is ALWAYS also within that same
            range, so the second clip never needs to change anything.
        defuzz_resolution : int
            Number of sample points used when discretizing the output
            universe for centroid defuzzification. Higher = more precise,
            slower.
        params : dict or None
            Optional explicit parameter set (see `get_default_params()`
            for the expected structure). If None, sensible defaults are
            used.
        """
        self.max_queue_for_scaling = max_queue_for_scaling
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.cycle_time = (
            cycle_time if cycle_time is not None
            else min_green_time + max_green_time
        )
        self.defuzz_resolution = defuzz_resolution

        self.params = params if params is not None else self.get_default_params()

    # ------------------------------------------------------------------
    # Parameter structure (this is what PSO/ACO tune)
    # ------------------------------------------------------------------

    def get_default_params(self):
        """
        Returns the default (hand-designed baseline) parameter set as
        a nested dict. This is the "human-readable" form; PSO/ACO work
        with the flat vector form instead (see `*_vector` methods),
        but both represent exactly the same information.

        Structure
        ---------
        {
          "mf_queue": {
              # Triangular membership functions for BOTH queue inputs
              # (shared shape for queue_1 and queue_2), each defined by
              # 3 points (a, b, c) meaning the triangle rises from a to
              # b (peak) and falls from b to c.
              "Low":    (a, b, c),
              "Medium": (a, b, c),
              "High":   (a, b, c),
          },
          "mf_green": {
              # Triangular membership functions for the green_time_1 output
              "Short":  (a, b, c),
              "Medium": (a, b, c),
              "Long":   (a, b, c),
          },
          "rule_weights": [w0, w1, ..., w8]
              # One weight in [0, 1] per rule in RULES, in the same order.
              # A weight of 0 effectively disables a rule; 1 is full strength.
        }
        """
        q_max = self.max_queue_for_scaling
        g_min, g_max = self.min_green_time, self.max_green_time
        g_mid = (g_min + g_max) / 2

        return {
            "mf_queue": {
                "Low":    (0,            0,            q_max * 0.35),
                "Medium": (q_max * 0.15, q_max * 0.5,  q_max * 0.85),
                "High":   (q_max * 0.65, q_max,        q_max),
            },
            "mf_green": {
                "Short":  (g_min,               g_min,               g_mid),
                "Medium": (g_min + (g_mid - g_min) * 0.3,
                           g_mid,
                           g_max - (g_max - g_mid) * 0.3),
                "Long":   (g_mid,               g_max,               g_max),
            },
            "rule_weights": [1.0] * len(self.RULES),
        }

    def get_param_bounds(self):
        """
        Returns (lower_bounds, upper_bounds) as flat numpy arrays,
        matching the layout of `params_to_vector`. PSO/ACO use these
        to initialize candidate solutions and clamp them to valid
        ranges.
        """
        q_max = self.max_queue_for_scaling
        g_min, g_max = self.min_green_time, self.max_green_time

        lower = []
        upper = []

        # 3 queue MFs x 3 points each, all within [0, q_max]
        for _ in range(3):
            lower += [0, 0, 0]
            upper += [q_max, q_max, q_max]

        # 3 green MFs x 3 points each, all within [g_min, g_max]
        for _ in range(3):
            lower += [g_min, g_min, g_min]
            upper += [g_max, g_max, g_max]

        # 9 rule weights, each within [0, 1]
        lower += [0.0] * len(self.RULES)
        upper += [1.0] * len(self.RULES)

        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    def params_to_vector(self, params=None):
        """Flatten a params dict (see `get_default_params`) into a 1D
        numpy array. If `params` is None, uses `self.params`."""
        p = params if params is not None else self.params

        vec = []
        for name in self.INPUT_SETS:
            vec.extend(p["mf_queue"][name])
        for name in self.OUTPUT_SETS:
            vec.extend(p["mf_green"][name])
        vec.extend(p["rule_weights"])

        return np.array(vec, dtype=float)

    def vector_to_params(self, vector):
        """Inverse of `params_to_vector`: unflatten a 1D numpy array
        back into the nested params dict structure."""
        vector = np.asarray(vector, dtype=float)
        idx = 0

        mf_queue = {}
        for name in self.INPUT_SETS:
            mf_queue[name] = tuple(vector[idx:idx + 3])
            idx += 3

        mf_green = {}
        for name in self.OUTPUT_SETS:
            mf_green[name] = tuple(vector[idx:idx + 3])
            idx += 3

        n_rules = len(self.RULES)
        rule_weights = list(vector[idx:idx + n_rules])
        idx += n_rules

        return {
            "mf_queue": mf_queue,
            "mf_green": mf_green,
            "rule_weights": rule_weights,
        }

    def get_default_vector(self):
        """Convenience: default params, already flattened."""
        return self.params_to_vector(self.get_default_params())

    def set_params_from_vector(self, vector):
        """Set this controller's active parameters from a flat vector.
        This is the main entry point PSO/ACO will call before evaluating
        a candidate solution."""
        self.params = self.vector_to_params(vector)

    # ------------------------------------------------------------------
    # Fuzzy logic core
    # ------------------------------------------------------------------

    @staticmethod
    def _triangular_membership(x, points):
        """
        Standard triangular membership function.

        points = (a, b, c): the membership rises linearly from 0 at a
        to 1 at b, then falls linearly from 1 at b to 0 at c.

        Handles "shoulder" shapes correctly (a==b, meaning the
        membership is already 1 at the left edge and only falls off
        to the right -- e.g. "Low" starting at the minimum of the
        input range; or b==c, the mirror case for e.g. "High" ending
        at the maximum). These are common and INTENTIONAL at the
        boundaries of the input universe -- they must NOT evaluate to
        0 at x==a or x==c in that case, since the peak (b) sits
        exactly on that boundary.
        """
        a, b, c = points
        if a == b == c:
            return 1.0 if x == a else 0.0

        if x <= a:
            # At or below the left edge: 1.0 only if the left edge
            # IS the peak (shoulder), otherwise 0.
            return 1.0 if a == b else 0.0
        if x >= c:
            # At or above the right edge: 1.0 only if the right edge
            # IS the peak (shoulder), otherwise 0.
            return 1.0 if b == c else 0.0
        if x == b:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        # b < x < c
        return (c - x) / (c - b) if c != b else 1.0

    def _fuzzify_queue(self, queue_length):
        """Returns membership degrees {Low: .., Medium: .., High: ..}
        for one queue length value."""
        mf = self.params["mf_queue"]
        return {
            name: self._triangular_membership(queue_length, mf[name])
            for name in self.INPUT_SETS
        }

    def compute_green_time(self, queue_length_1, queue_length_2):
        """
        Main entry point: given both roads' current queue lengths,
        returns (green_time_1, green_time_2) in seconds.

        This is a Mamdani inference pipeline:
            1. Fuzzify both inputs (membership degrees per set)
            2. Evaluate each rule's firing strength (AND = min),
               scaled by that rule's weight
            3. Aggregate each output set's activation (OR = max
               across rules that point to the same output set)
            4. Defuzzify via centroid over the aggregated output
               membership function

        NOTE on defuzzification method (important, do not change
        without team discussion):
        This implements the CLASSIC Mamdani pipeline -- clip each
        rule's output shape at its firing strength, take the UNION
        (max) of all clipped shapes across rules, THEN compute the
        centroid (center of gravity) of that single combined shape.
        This matches the "Graphical representation of Mamdani method
        with singleton input" diagram from the course slides (clip ->
        aggregate via min-block union -> centroid on the combined C').

        This is DIFFERENT from, and NOT combined with, the "Center
        Average" (CA) method also shown in the slides:
            y* = sum(y_bar_l * w_l) / sum(w_l)
        Center Average skips aggregation entirely -- it just takes a
        weighted average of each rule's output CENTER POINT directly,
        with no shape-building or union step at all.

        These two methods are ALTERNATIVES to each other, never used
        together. We use full aggregation + centroid here because the
        project spec explicitly requires "centroid" defuzzification.
        If a future revision switches to Center Average for speed,
        the aggregation step (`output_activation` union via max) must
        be REMOVED, not extended -- mixing the two would double-count
        the aggregation logic and does not correspond to either
        method in the slides.
        """
        deg_1 = self._fuzzify_queue(queue_length_1)
        deg_2 = self._fuzzify_queue(queue_length_2)

        # Step 2 & 3: evaluate rules, aggregate per output set
        output_activation = {name: 0.0 for name in self.OUTPUT_SETS}
        weights = self.params["rule_weights"]

        for i, (set_1, set_2, out_set) in enumerate(self.RULES):
            firing_strength = min(deg_1[set_1], deg_2[set_2]) * weights[i]
            output_activation[out_set] = max(output_activation[out_set], firing_strength)

        # Step 4: centroid defuzzification
        green_1 = self._centroid_defuzzify(output_activation)
        green_1 = float(np.clip(green_1, self.min_green_time, self.max_green_time))

        green_2 = self.cycle_time - green_1
        green_2 = float(np.clip(green_2, self.min_green_time, self.max_green_time))

        return green_1, green_2

    def _centroid_defuzzify(self, output_activation):
        """
        Centroid (center of gravity) defuzzification over the
        aggregated output fuzzy set, computed numerically by sampling
        the output universe.
        """
        mf = self.params["mf_green"]
        x_samples = np.linspace(
            self.min_green_time, self.max_green_time, self.defuzz_resolution
        )

        aggregated = np.zeros_like(x_samples)
        for name in self.OUTPUT_SETS:
            activation = output_activation[name]
            if activation <= 0:
                continue
            mf_values = np.array([
                min(self._triangular_membership(x, mf[name]), activation)
                for x in x_samples
            ])
            aggregated = np.maximum(aggregated, mf_values)

        total_area = np.sum(aggregated)
        if total_area == 0:
            # No rule fired at all (shouldn't normally happen since the
            # rule base covers all input combinations) -- fall back to
            # the midpoint of the output range as a safe default.
            return (self.min_green_time + self.max_green_time) / 2

        centroid = np.sum(aggregated * x_samples) / total_area
        return centroid