# Hourly Centralized Maskable PPO

This is the formal, minimally structured PPO baseline.

- One policy transition advances exactly one physical hour.
- At every hour the policy observes the current physical state and the shared
  168 h structured future summary.
- The policy directly outputs the factorised per-vessel dispatch action.
- Legal-action masks are the only action-side aid.
- There is no event trigger, high-level goal, rule executor, Greedy default,
  residual action, behaviour cloning, or agent-selected well rate.
- Reward is scaled negative realised economic cost. The common compact trip
  cleanup value is added to the final transition, which is marked terminal.

Formal training:

```powershell
python -m sim.control.hourly_ppo.train_hourly_ppo `
  --episode-hours 720 `
  --forecast-context-hours 168 `
  --future-summary-windows-h 168 `
  --gamma 1 `
  --max-simulator-hour-steps 9505319 `
  --validation-seeds 8100001 8100002 8100003 `
  --device cpu
```

Deterministic evaluation:

```powershell
python -m sim.control.hourly_ppo.evaluate_hourly_ppo `
  --run-dir logs\hourly_ppo\YOUR_RUN `
  --model best `
  --seeds 9000001 9000002 9000003
```
