# BC warm-start + PPO — bc_phase1_720h_inj80_weather_ts150000

Generated: 2026-07-06T19:28:50

bc_episodes=30, bc_epochs=20, timesteps=150000, injection_reward=80.0

```
## After BC (before PPO)
idle                 storage=  2.2%  loss= 56.3%
greedy_shuttle       storage= 82.0%  loss=  0.7%
ppo_stochastic       storage= 81.3%  loss=  1.8%
ppo_deterministic    storage= 82.7%  loss=  2.6%

## After PPO fine-tune
idle                 storage=  2.2%  loss= 56.3%
greedy_shuttle       storage= 82.0%  loss=  0.7%
ppo_stochastic       storage= 58.0%  loss= 19.9%
ppo_deterministic    storage= 58.1%  loss= 23.5%
```
