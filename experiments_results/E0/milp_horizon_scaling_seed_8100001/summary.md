# MILP horizon-scaling implementation diagnostic

## Protocol

- Role: implementation diagnostic; not a formal paper comparison.
- Scenario: one fixed `unified_window_v1` sample, seed `8100001`.
- Boundary: one 888 h sampled scenario (720 h online period + 168 h forecast
  context); every solve starts from the same initial state.
- Planning horizons: 12, 24, 48, 72, 96 and 168 h.
- Objective: environment-aligned economic cost plus common terminal cleanup.
- Well control: shared continuous automatic-maximum rule; no controller well
  action.
- Warm start: Greedy only.
- Solver: CPLEX 12.10, deterministic parallel mode, 4 threads, 300 s limit for
  every horizon.

## Results

| Horizon (h) | Variables | Constraints | Presolved binaries | Solve time (s) | Status | Gap | Greedy start (EUR) | MILP incumbent (EUR) | Improvement vs Greedy | Exact replay |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|:---:|
| 12 | 2,407 | 4,305 | 277 | 0.30 | Optimal | 0.00% | 172,375.82 | 172,375.82 | 0.00% | Yes |
| 24 | 4,711 | 8,362 | 724 | 1.25 | Optimal | 0.00% | 226,111.06 | 216,147.70 | 4.41% | Yes |
| 48 | 9,319 | 16,486 | 2,136 | 53.66 | Optimal | 0.00% | 262,805.84 | 260,315.00 | 0.95% | Yes |
| 72 | 13,927 | 24,610 | 4,460 | 143.13 | Optimal | 0.00% | 309,283.69 | 308,868.55 | 0.13% | Yes |
| 96 | 18,535 | 32,734 | 7,211 | 301.01 | Integer feasible | 9.77% | 382,010.97 | 381,595.83 | 0.11% | Yes |
| 168 | 32,359 | 57,106 | 15,038 | 302.03 | Integer feasible | 7.69% | 526,082.39 | 524,006.69 | 0.39% | Yes |

The replay-minus-solver objective error was at most
`2.15e-6 EUR` across all six horizons.

## Interpretation

1. The shared MILP is feasible and simulator-consistent at every tested
   horizon. All six Greedy starts were accepted, all returned incumbents were
   executable, and all replays matched the MILP prediction.
2. The model reached a proven optimum through 72 h. The practical scaling
   transition occurred between 72 and 96 h for this fixed scenario and
   300 s budget.
3. The 168 h solve improved the Greedy start by `2,075.70 EUR`, so CPLEX was
   not merely returning an unchanged warm start.
4. At 168 h, CPLEX spent the entire budget processing cuts at the root node:
   it reduced the gap from 136.93% at warm-start acceptance to 7.69%, but did
   not enter branch-and-bound. This points to formulation size and relaxation
   strength as the immediate bottleneck.
5. These results do not prove that every long-horizon constraint is bug-free,
   but they provide strong evidence against a fundamental feasibility,
   automatic-well, objective-accounting, or replay implementation defect.
