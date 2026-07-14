"""
cost_function.py

Implements the project's cost function:

    C = alpha * W + beta * Q + gamma * S

where:
    W = average waiting time     (from TrafficEnv.get_metrics())
    Q = average queue length     (from TrafficEnv.get_metrics())
    S = number of stops          (from TrafficEnv.get_metrics())

This is the single number PSO and ACO are trying to MINIMIZE.
"""


DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 0.1
# NOTE: S (num_stops) is typically a much larger raw number than W or Q
# over a long simulation, so gamma starts smaller by default to keep
# the three terms on a comparable scale. Tune these three weights
# together when running experiments in Phase 4.


def compute_cost(metrics, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA):
    """
    Compute the scalar cost C = alpha*W + beta*Q + gamma*S.

    Parameters
    ----------
    metrics : dict
        The dict returned by TrafficEnv.get_metrics(), i.e. must
        contain the keys "average_waiting_time", "average_queue_length",
        and "num_stops".
    alpha, beta, gamma : float
        Weighting coefficients for W, Q, and S respectively.

    Returns
    -------
    float
        The scalar cost value. Lower is better.
    """
    W = metrics["average_waiting_time"]
    Q = metrics["average_queue_length"]
    S = metrics["num_stops"]

    return alpha * W + beta * Q + gamma * S


def evaluate_controller(controller, env_factory, num_steps, alpha=DEFAULT_ALPHA,
                         beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA, decision_interval=1):
    """
    Convenience wrapper: run a full simulation using the given fuzzy
    controller, then compute its cost. This is the exact function
    PSO/ACO will call once per candidate solution.

    Parameters
    ----------
    controller : FuzzyController
        An already-configured controller (i.e. `set_params_from_vector`
        has already been called with the candidate solution).
    env_factory : callable
        A zero-argument function that returns a freshly-reset
        TrafficEnv instance, e.g. `lambda: TrafficEnv(arrival_rate_1=0.4,
        arrival_rate_2=0.2, departure_rate=1.0, seed=42)`.
        Using a factory (rather than passing an env directly) makes it
        easy to guarantee a clean environment per evaluation, and to
        control random seeds for fair comparisons across candidates.
    num_steps : int
        How many simulation ticks to run per evaluation.
    alpha, beta, gamma : float
        Cost function weights, see `compute_cost`.
    decision_interval : int
        How often (in ticks) the controller re-evaluates and can change
        which road is green. 1 means it re-decides every tick. Larger
        values simulate a minimum green-time commitment, which is more
        realistic (real lights don't flip every second) -- Phase 3 may
        want to set this higher (e.g. equal to typical green_time).

    Returns
    -------
    float
        The cost of this controller on this simulation run.
    """
    env = env_factory()
    env.reset()

    current_green = 1
    ticks_since_decision = 0

    for t in range(num_steps):
        if ticks_since_decision % decision_interval == 0:
            green_1_time, green_2_time = controller.compute_green_time(
                env.queue_length_1, env.queue_length_2
            )
            # Simple policy: whichever road's computed green time is
            # longer gets priority this decision window. See module
            # docstring in fuzzy_controller.py for how green_1/green_2
            # are derived from the single fuzzy output.
            current_green = 1 if green_1_time >= green_2_time else 2

        env.step(current_green)
        ticks_since_decision += 1

    metrics = env.get_metrics()
    return compute_cost(metrics, alpha=alpha, beta=beta, gamma=gamma)