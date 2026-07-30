# SLURM accounting

| Job | Purpose | State | Exit code | Elapsed |
|---|---|---|---|---:|
| 34211 | one-seed smoke evaluation | COMPLETED | 0:0 | 00:00:08 |
| 34212_0 | formal evaluation, model seed 0 | COMPLETED | 0:0 | 00:00:42 |
| 34212_1 | formal evaluation, model seed 1 | COMPLETED | 0:0 | 00:00:44 |
| 34212_2 | formal evaluation, model seed 2 | COMPLETED | 0:0 | 00:00:45 |

Preceding failed attempt:

| Job | Purpose | State | Exit code | Note |
|---|---|---|---|---|
| 34209 | smoke attempt for run01 | FAILED | 1:0 | stopped before model evaluation because `pytest` was unavailable |
| 34210 | dependent formal array for run01 | CANCELLED | 0:0 | dependency never satisfied; no array task ran |
