# Reward alignment check (step-1 shaping)

Generated: 2026-07-06T16:19:37 | Phase 1, seed=1.

Cumulative episode reward earned by fixed policies. If shaping fixes the
objective, greedy_shuttle (stores more) should out-earn idle at every horizon.

| horizon | shaping | idle reward | greedy reward | objective rewards |
|---|---|---|---|---|
| 168h | off | -230 | -336 | IDLE (misaligned) |
| 168h | on(80/t) | -38 | 620 | greedy (aligned) |
| 720h | off | -5,776 | -1,525 | greedy (aligned) |
| 720h | on(80/t) | -5,584 | 6,762 | greedy (aligned) |