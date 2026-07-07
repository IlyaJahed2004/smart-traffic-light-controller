"""
Discrete-time simulation of a single two-road intersection.

This is Layer 1 of the project: it models car arrivals, queues, and
departures, and exposes the raw data (queue lengths, waiting times,
stop counts) that the fuzzy controller and cost function need.

The simulation does NOT decide green-light durations itself -- that
decision is passed in externally at each `step()` call. This keeps
the simulation decoupled from the controller, so we can plug in:
    - a fixed-timer controller (for testing this module)
    - a fuzzy controller (Phase 2)
    - a PSO/ACO-tuned fuzzy controller (Phase 3)
without changing this file.
"""

import numpy as np


class Car:
    """A single car in a queue. Tracks when it arrived and whether it
    has had to come to a stop (used for the S term in the cost function)."""

    def __init__(self, arrival_time):
        self.arrival_time = arrival_time
        self.has_stopped = False


class TrafficEnv:
    """
    Discrete-time two-road intersection simulator.

    Parameters
    ----------
    arrival_rate_1 : float
        Average number of cars arriving per time step on Road 1
        (Poisson lambda). E.g. 0.3 means ~0.3 cars/step on average.
    arrival_rate_2 : float
        Average number of cars arriving per time step on Road 2.
    departure_rate : float
        Max number of cars that can leave the queue per time step
        while that road has a green light (discharge rate).
    time_step : float
        Duration, in seconds, represented by one simulation step.
        Purely informational unless you want to convert step counts
        to real seconds later (e.g. for reporting).
    clearance_time : int
        Number of steps of "all-red" inserted whenever the green
        light switches from one road to the other. Defaults to 0
        (instant switch, no clearance). Kept flexible for later use.
    max_queue : int or None
        Optional cap on queue length (models a finite road). None
        means unbounded.
    seed : int or None
        Random seed for reproducibility.

    Usage
    -----
        env = TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2,
                          departure_rate=1.0)
        env.reset()
        for t in range(num_steps):
            green_road = my_controller(env.queue_length_1, env.queue_length_2)
            env.step(green_road)
        metrics = env.get_metrics()
    """

    def __init__(
        self,
        arrival_rate_1=0.3,
        arrival_rate_2=0.3,
        departure_rate=1.0,
        time_step=1.0,
        clearance_time=0,
        max_queue=None,
        seed=None,
    ):
        self.arrival_rate_1 = arrival_rate_1
        self.arrival_rate_2 = arrival_rate_2
        self.departure_rate = departure_rate
        self.time_step = time_step
        self.clearance_time = clearance_time
        self.max_queue = max_queue

        self._rng = np.random.default_rng(seed)

        self.reset()

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the environment to an empty, time-zero state."""
        self.current_time = 0

        self.queue_1 = []  # list[Car] waiting on road 1
        self.queue_2 = []  # list[Car] waiting on road 2

        self.current_green = None  # 1, 2, or None (during clearance)
        self._clearance_remaining = 0
        self._previous_green = None

        # Metrics accumulated over the whole run
        self.completed_wait_times = []  # waiting time of every car that departed
        self.stop_count = 0             # number of cars that had to stop at least once
        self.total_cars_arrived = 0
        self.total_cars_departed = 0

        # Per-step history (useful for plots / debugging)
        self.history = {
            "time": [],
            "queue_1": [],
            "queue_2": [],
            "green": [],
        }

        return self._get_state()

    def _get_state(self):
        """Return the observable state the controller is allowed to see."""
        return {
            "queue_length_1": len(self.queue_1),
            "queue_length_2": len(self.queue_2),
            "current_green": self.current_green,
        }

    # ------------------------------------------------------------------
    # Step function
    # ------------------------------------------------------------------

    def step(self, green_road):
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        green_road : int (1 or 2)
            Which road the controller wants to be green during this step.
            If this differs from the previous step's green road and
            `clearance_time` > 0, a clearance period is inserted first
            (during which neither road discharges cars).

        Returns
        -------
        state : dict
            The new observable state (queue lengths etc.)
        """
        assert green_road in (1, 2), "green_road must be 1 or 2"

        # --- Handle clearance (yellow/all-red) period ---
        if self.clearance_time > 0 and green_road != self._previous_green \
                and self._previous_green is not None:
            self._clearance_remaining = self.clearance_time

        if self._clearance_remaining > 0:
            effective_green = None  # no discharge this step
            self._clearance_remaining -= 1
        else:
            effective_green = green_road

        self.current_green = effective_green
        self._previous_green = green_road

        # --- Arrivals (Poisson process per road) ---
        new_arrivals_1 = self._rng.poisson(self.arrival_rate_1)
        new_arrivals_2 = self._rng.poisson(self.arrival_rate_2)

        for _ in range(new_arrivals_1):
            if self.max_queue is None or len(self.queue_1) < self.max_queue:
                self.queue_1.append(Car(self.current_time))
                self.total_cars_arrived += 1

        for _ in range(new_arrivals_2):
            if self.max_queue is None or len(self.queue_2) < self.max_queue:
                self.queue_2.append(Car(self.current_time))
                self.total_cars_arrived += 1

        # --- Mark stops: any car waiting in a queue that is NOT
        #     currently discharging counts as "stopped" this step ---
        if effective_green != 1:
            for car in self.queue_1:
                if not car.has_stopped:
                    car.has_stopped = True
                    self.stop_count += 1
        if effective_green != 2:
            for car in self.queue_2:
                if not car.has_stopped:
                    car.has_stopped = True
                    self.stop_count += 1

        # --- Departures (only the green road discharges) ---
        if effective_green == 1:
            self._discharge(self.queue_1)
        elif effective_green == 2:
            self._discharge(self.queue_2)

        # --- Record history ---
        self.history["time"].append(self.current_time)
        self.history["queue_1"].append(len(self.queue_1))
        self.history["queue_2"].append(len(self.queue_2))
        self.history["green"].append(effective_green)

        self.current_time += 1

        return self._get_state()

    def _discharge(self, queue):
        """Remove up to `departure_rate` cars from the front of a queue,
        recording their waiting time."""
        num_to_discharge = int(self.departure_rate)
        # Handle fractional departure rates probabilistically
        frac = self.departure_rate - num_to_discharge
        if self._rng.random() < frac:
            num_to_discharge += 1

        for _ in range(min(num_to_discharge, len(queue))):
            car = queue.pop(0)
            wait_time = (self.current_time - car.arrival_time) * self.time_step
            self.completed_wait_times.append(wait_time)
            self.total_cars_departed += 1

    # ------------------------------------------------------------------
    # Metrics (used later by cost_function.py)
    # ------------------------------------------------------------------

    def get_metrics(self):
        """
        Summarize the run so far into the raw quantities the cost
        function needs: average waiting time (W), average queue
        length (Q), and number of stops (S).
        """
        avg_wait = float(np.mean(self.completed_wait_times)) \
            if self.completed_wait_times else 0.0

        avg_queue = float(np.mean(
            np.array(self.history["queue_1"]) + np.array(self.history["queue_2"])
        )) if self.history["time"] else 0.0

        return {
            "average_waiting_time": avg_wait,
            "average_queue_length": avg_queue,
            "num_stops": self.stop_count,
            "total_cars_arrived": self.total_cars_arrived,
            "total_cars_departed": self.total_cars_departed,
        }

    @property
    def queue_length_1(self):
        return len(self.queue_1)

    @property
    def queue_length_2(self):
        return len(self.queue_2)