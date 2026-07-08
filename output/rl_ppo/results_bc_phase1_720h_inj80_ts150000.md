# BC warm-start + PPO — bc_phase1_720h_inj80_ts150000

Generated: 2026-07-06T18:51:51

bc_episodes=30, bc_epochs=20, timesteps=150000, injection_reward=80.0

```
## After BC (before PPO)
idle                 storage=  2.2%  loss= 56.3%
greedy_shuttle       storage= 82.0%  loss=  0.7%
ppo_stochastic       storage= 79.2%  loss=  0.4%
ppo_deterministic    storage= 81.2%  loss=  2.2%

## After PPO fine-tune
idle                 storage=  2.2%  loss= 56.3%
greedy_shuttle       storage= 82.0%  loss=  0.7%
ppo_stochastic       storage= 66.1%  loss=  9.1%
ppo_deterministic    storage= 76.1%  loss=  5.5%
```
