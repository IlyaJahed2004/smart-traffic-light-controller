# Phase 1 — Traffic Simulation

**Code:** `src/simulation/traffic_env.py`
**Test it:** `python experiments/test_traffic_env.py`
**Status:** ✅ Done

## What this is, in plain words

This is a fake little world with **one intersection and two roads**. Cars randomly show up on each road and wait in line. Every "tick" (one time step), you tell the simulation which road currently has the green light, and it:
- lets a few cars through on the green road
- adds any new cars that just arrived
- keeps track of how long cars waited and how many had to stop

This module **does not decide which road should be green** — that's the fuzzy controller's job (Phase 2). This module is just the "world" that the controller will be tested against.

## The settings (config variables), explained simply

| Variable | What it means | Example |
|---|---|---|
| `arrival_rate_1`, `arrival_rate_2` | On average, how many new cars show up per tick on each road. Not exact — some ticks get 0 cars, some get 1, occasionally more. | `0.4` ≈ "about 4 cars every 10 ticks" |
| `departure_rate` | How many cars can drive through per tick *while their light is green*. A green light doesn't clear the whole queue instantly — only this many cars get through each tick. | `1.0` ≈ "1 car leaves per tick on green" |
| `time_step` | How many real-world seconds one tick represents. Just a label for converting ticks into real time later. | `1.0` = 1 tick is 1 second |
| `clearance_time` | How many ticks the light stays "all red" (like a yellow light pause) right after switching which road is green. `0` means instant switching, no pause. | `2` = 2 ticks of nobody moving after a switch |
| `max_queue` | The most cars a road can hold before it's "full." `None` means unlimited. | `None` (default) |
| `seed` | A number that makes the "randomness" repeatable — same seed = same sequence of arrivals every time you run it. Useful for testing and comparing fairly. | `42` |

## The methods, explained simply

### `reset()`
**"Clear the board."** Empties both roads, sets the clock back to 0, wipes all counters. Always call this before starting a new run.

### `step(green_road)`
**"Let one tick pass."** You pass in `1` or `2` — whichever road you want to be green right now. Inside, it does 3 things in order:
1. **New cars arrive** (randomly, on both roads)
2. **Cars on the green road drive off** (up to `departure_rate` of them)
3. **Everyone else keeps waiting**, and gets marked as "stopped" if it's their first tick waiting

You call this once per tick, in a loop — like pressing a stopwatch button over and over.

### `get_metrics()`
**"Give me the report card."** Call this after running for a while. It hands back:
- `average_waiting_time` — on average, how long did cars wait before finally driving off? (This is **W** in the cost function)
- `average_queue_length` — on average, how many cars were sitting in line at any given moment? (This is **Q**)
- `num_stops` — how many cars, in total, had to come to a stop at least once? (This is **S**)
- `total_cars_arrived` / `total_cars_departed` — just for double-checking nothing got lost or duplicated

### `queue_length_1` / `queue_length_2`
**"How many cars are waiting right now?"** Quick way to peek at each road's current line length. This is exactly what the Phase 2 fuzzy controller will look at to decide which road should get green next.

## A tiny example

```python
from simulation.traffic_env import TrafficEnv

env = TrafficEnv(arrival_rate_1=0.4, arrival_rate_2=0.2, departure_rate=1.0, seed=42)
env.reset()

green = 1
for t in range(2000):
    if t % 10 == 0:               # dumb rule for testing: switch every 10 ticks
        green = 2 if green == 1 else 1
    env.step(green)

print(env.get_metrics())
# {'average_waiting_time': 5.47, 'average_queue_length': 3.06,
#  'num_stops': 627, 'total_cars_arrived': 1119, 'total_cars_departed': 1118}
```

Or just run the ready-made script instead of typing this:

```bash
python experiments/test_traffic_env.py
```

## Important things to know

- **A car arriving during a green tick can leave in that same tick** — arrivals are processed before departures, so nobody has to wait a full extra tick just from bad timing.
- **Multiple cars can arrive in a single tick** — since arrivals are random, it's not always exactly 0 or 1; a queue can jump by 2 or 3 in one tick sometimes.
- **Only `departure_rate` cars leave per tick, no matter how many are waiting** — so a long queue takes several ticks to clear, even with the light green the whole time.

## What Phase 2 needs from this file

The fuzzy controller will, each tick:
1. Look at `env.queue_length_1` and `env.queue_length_2`
2. Decide a `green_road` (1 or 2) — and eventually, how *long* to keep it green
3. Call `env.step(green_road)`
4. At the end, call `env.get_metrics()` to compute the cost function `C = α·W + β·Q + γ·S`

Nothing in this file needs to change for that to work — the interface is already what Phase 2 expects.