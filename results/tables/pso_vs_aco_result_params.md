# PSO vs ACO — tuned parameter values

Baseline cost (default vector): **73.6823**


## Baseline
```json
{
  "mf_queue": {
    "Low": [
      0.0,
      0.0,
      7.0
    ],
    "Medium": [
      3.0,
      10.0,
      17.0
    ],
    "High": [
      13.0,
      20.0,
      20.0
    ]
  },
  "mf_green": {
    "Short": [
      5.0,
      5.0,
      25.0
    ],
    "Medium": [
      11.0,
      25.0,
      39.0
    ],
    "Long": [
      25.0,
      45.0,
      45.0
    ]
  },
  "rule_weights": {
    "R0: IF q1=Low AND q2=Low THEN green=Medium": 1.0,
    "R1: IF q1=Low AND q2=Medium THEN green=Short": 1.0,
    "R2: IF q1=Low AND q2=High THEN green=Short": 1.0,
    "R3: IF q1=Medium AND q2=Low THEN green=Long": 1.0,
    "R4: IF q1=Medium AND q2=Medium THEN green=Medium": 1.0,
    "R5: IF q1=Medium AND q2=High THEN green=Short": 1.0,
    "R6: IF q1=High AND q2=Low THEN green=Long": 1.0,
    "R7: IF q1=High AND q2=Medium THEN green=Long": 1.0,
    "R8: IF q1=High AND q2=High THEN green=Medium": 1.0
  }
}
```

## PSO

- Best cost: **68.1014**
- Improvement over baseline: **+5.5809**
- Runtime: 611.54s

```json
{
  "mf_queue": {
    "Low": [
      0.0,
      0.0,
      5.5885
    ],
    "Medium": [
      0.0,
      7.4238,
      18.4083
    ],
    "High": [
      16.7957,
      20.0,
      20.0
    ]
  },
  "mf_green": {
    "Short": [
      5.0,
      5.0,
      15.0036
    ],
    "Medium": [
      5.0,
      9.1123,
      36.1878
    ],
    "Long": [
      23.9845,
      43.585,
      45.0
    ]
  },
  "rule_weights": {
    "R0: IF q1=Low AND q2=Low THEN green=Medium": 0.9768,
    "R1: IF q1=Low AND q2=Medium THEN green=Short": 0.9491,
    "R2: IF q1=Low AND q2=High THEN green=Short": 0.9383,
    "R3: IF q1=Medium AND q2=Low THEN green=Long": 0.998,
    "R4: IF q1=Medium AND q2=Medium THEN green=Medium": 1.0,
    "R5: IF q1=Medium AND q2=High THEN green=Short": 0.986,
    "R6: IF q1=High AND q2=Low THEN green=Long": 1.0,
    "R7: IF q1=High AND q2=Medium THEN green=Long": 0.9978,
    "R8: IF q1=High AND q2=High THEN green=Medium": 1.0
  }
}
```

## ACO

- Best cost: **72.5857**
- Improvement over baseline: **+1.0966**
- Runtime: 744.89s

```json
{
  "mf_queue": {
    "Low": [
      1.5,
      1.5,
      17.5
    ],
    "Medium": [
      0.5,
      3.5,
      15.5
    ],
    "High": [
      12.5,
      14.5,
      14.5
    ]
  },
  "mf_green": {
    "Short": [
      6.0,
      6.0,
      34.0
    ],
    "Medium": [
      22.0,
      34.0,
      40.0
    ],
    "Long": [
      22.0,
      26.0,
      44.0
    ]
  },
  "rule_weights": {
    "R0: IF q1=Low AND q2=Low THEN green=Medium": 0.375,
    "R1: IF q1=Low AND q2=Medium THEN green=Short": 0.775,
    "R2: IF q1=Low AND q2=High THEN green=Short": 0.625,
    "R3: IF q1=Medium AND q2=Low THEN green=Long": 0.225,
    "R4: IF q1=Medium AND q2=Medium THEN green=Medium": 0.075,
    "R5: IF q1=Medium AND q2=High THEN green=Short": 0.875,
    "R6: IF q1=High AND q2=Low THEN green=Long": 0.375,
    "R7: IF q1=High AND q2=Medium THEN green=Long": 0.175,
    "R8: IF q1=High AND q2=High THEN green=Medium": 0.125
  }
}
```
