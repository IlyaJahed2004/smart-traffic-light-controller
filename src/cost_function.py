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
                         beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA):
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
 
    Returns
    -------
    float
        The cost of this controller on this simulation run.
 
    How the light actually switches (IMPORTANT)
    ---------------------------------------------
    `controller.compute_green_time(q1, q2)` returns a PLAN: "if I had
    a full `cycle_time`-second cycle right now, road 1 should get
    `green_1` seconds and road 2 should get `green_2` seconds"
    (green_1 + green_2 == cycle_time, always).
 
    This function HONORS that plan literally:
        1. Ask the controller for a plan (green_1, green_2) based on
           the CURRENT queue lengths.
        2. Hold road 1 green for round(green_1) ticks in a row.
        3. Then hold road 2 green for round(green_2) ticks in a row.
        4. Only THEN ask the controller again for a new plan (queues
           have changed by now, since cars kept arriving/leaving
           during those ticks).
 
    This matches how a real traffic light works -- it does NOT flip
    back and forth every tick. An earlier version of this function
    re-asked the controller every single tick and just used
    `green_1 >= green_2` as a per-tick tie-break, which caused the
    light to flicker unrealistically (observed flipping multiple
    times within a handful of ticks). That version is deprecated;
    this one replaces it.
    """
    env = env_factory()
    env.reset()
 
    ticks_run = 0
    while ticks_run < num_steps:
        # Ask for a fresh plan based on current queue lengths
        green_1_time, green_2_time = controller.compute_green_time(
            env.queue_length_1, env.queue_length_2
        )
 
        # Convert seconds -> whole ticks, at least 1 tick each so a
        # road is never skipped entirely even if its share rounds to 0
        green_1_ticks = max(1, round(green_1_time))
        green_2_ticks = max(1, round(green_2_time))
 
        # Run road 1's green phase
        for _ in range(green_1_ticks):
            if ticks_run >= num_steps:
                break
            env.step(1)
            ticks_run += 1
 
        # Run road 2's green phase
        for _ in range(green_2_ticks):
            if ticks_run >= num_steps:
                break
            env.step(2)
            ticks_run += 1
 
    metrics = env.get_metrics()
    return compute_cost(metrics, alpha=alpha, beta=beta, gamma=gamma)
