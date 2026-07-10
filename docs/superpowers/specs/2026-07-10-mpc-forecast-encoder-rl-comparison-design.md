# MPC-Teacher Forecast-Encoding RL Comparison

## Objective

Compare three 720-hour MaskablePPO agents under one controlled experimental protocol:

1. **Current-state PPO** receives the operational state and current global weather, but no future disturbance sequence.
2. **Flat-forecast PPO** receives the same current state plus a flattened `168 x 9` forecast.
3. **TCN-forecast PPO** receives the same current state plus the same `168 x 9` forecast encoded to a 64-dimensional latent by a small temporal convolutional network.

All three agents use the same replay-validated `RollingNativeMpcController` demonstrations for behaviour-cloning warm-start and the same decaying kickstarting anchor during PPO. The experiment isolates the effect of forecast availability and forecast representation; it does not simultaneously change the reward.

## Fixed Experimental Setting

- Scenario: `northern_lights_phase1_3vessels`.
- Yara buffer: the registered 15,000 t value; no runtime override.
- RL episode: 720 one-hour steps.
- MPC replan interval: 24 h.
- MPC lookahead: 168 h.
- Weather process: one shared global speed factor, resampled every 24 h (`block`).
- Capture noise, capture outage, high-output windows, well maintenance, initial inventories, and warm-start settings: identical across all variants.
- Reward: existing `vent_first` reward with venting as the primary term, overflow risk as dense warning, and operating cost as the secondary per-step term.
- Dispatch: partial-load dispatch remains legal.
- Action masking: identical MaskablePPO masks for all variants.

The first comparison deliberately leaves downstream-inventory potential shaping out. Adding it now would confound forecast-architecture effects with reward effects. End-of-episode inventory remains an evaluation metric and can be a later reward ablation if the trained agents defer work without venting.

## Information Contract

The teacher and forecast-enabled students must receive the same disturbance trajectory. A shared forecast view exposes the sampled scenario instead of allowing the teacher to read hidden future state through one path while the student receives a different summary through another.

The initial experiment is an oracle-forecast experiment: the next 168 hourly values are exact values from the sampled scenario. This matches the current `RollingNativeMpcController`, which evaluates candidates by rolling a copied environment through the sampled future. A later robustness experiment may replace the shared view with noisy forecasts, but both MPC and RL must then consume that same noisy forecast.

### Current-state branch

The current-state vector contains the existing normalized physical state:

- weekly clock and total in-transit fill;
- emitter fill, current capture, and current availability;
- vessel cargo, berth/location, destination, and sailing progress;
- terminal fill and berth availability;
- current well injection, injectivity, and availability;
- reservoir pressure margin;
- current global weather factor and current normalized travel times.

For the three-vessel scenario this is expected to be approximately 51 dimensions after separating current weather from future summaries. Feature names and the exact shape are generated and tested rather than hard-coded into training logic.

### Forecast branch

The forecast covers future hours `t+1..t+168` and has shape `[168, 9]`. The current hour `t` is already represented by the current-state branch, so it is not duplicated in the forecast:

1. three normalized emitter capture-rate channels;
2. three binary emitter-availability channels;
3. one binary well-availability channel;
4. one normalized well-injectivity channel;
5. one global vessel-speed-factor channel, stored at channel index 8 (the ninth channel).

Emitter capture is normalized by the corresponding emitter maximum production rate. Binary availability is retained even though outages also reduce capture, because it distinguishes an outage from ordinary low capture. Weather is global and is therefore not duplicated per vessel; current travel times remain in the state branch. In 24 h block-weather mode, channel 8 is piecewise constant across each physical weather block and exposes the remaining current block followed by future blocks.

The environment returns forecast tensors in time-major order `[168, 9]`. The TCN extractor transposes each batch to PyTorch `Conv1d` order `[batch, 9, 168]` without changing the underlying data.

All values must be finite and have stable channel order. The forecast metadata records channel names, horizon, scenario configuration, and normalization constants.

## Episode-Tail Handling

A 720 h rollout still needs a full 168 h forecast at hour 719, and SB3 timeout bootstrapping needs the same full forecast in the terminal observation at hour 720. Scenario generation therefore produces 889 h of disturbances while the RL environment truncates after 720 h.

MPC demonstration environments run for 889 h, but only their first 720 state-action pairs are collected. Consequently, every collected MPC decision and the RL terminal observation have a full 168 h lookahead. RL environments truncate at 720 h while reading their forecast from the same 889 h sampled trajectory. The extra 169 h is forecast and timeout-bootstrap context only and is excluded from episode KPIs and PPO timesteps.

The timeout observation is a genuine hour-720 state: its current emitter availability, well availability, and injectivity come from the hour-720 scenario without mutating the ended native environment, and its future forecast starts at hour 721.

This separation prevents the MPC horizon from shrinking near the artificial episode boundary and avoids padding the forecast with fabricated values.

## Policy Variants

### Variant A: current-state PPO

- Input: current-state branch only.
- Purpose: measure performance without future disturbance information.
- Teacher demonstrations are still used for BC and kickstarting, so this variant also quantifies the effect of privileged-teacher label ambiguity.

### Variant B: flat-forecast PPO

- Input: current state concatenated with the flattened 1,512 forecast values.
- Extractor: ordinary MLP.
- Purpose: measure the value of the full oracle forecast without a temporal inductive bias.

### Variant C: TCN-forecast PPO

- State extractor: small MLP producing 64 features.
- Forecast extractor: three one-dimensional convolutional layers over the time axis, retaining temporal position, followed by a linear projection to 64 features.
- Combined feature: 128 dimensions before the PPO actor and critic heads.
- No global average pooling, because it would erase whether an outage occurs tomorrow or near the end of the horizon.

The forecast encoder consumes the same raw `[168, 9]` tensor as Variant B. It reduces the effective policy representation, not the amount of information supplied by the environment.

## Demonstration Pipeline

MPC demonstrations are generated once, offline, before GPU training. Each cached record contains:

- current-state observation;
- full `[168, 9]` forecast;
- flattened native action;
- action legality mask;
- environment seed and simulation hour;
- selected MPC candidate and replay-validation status;
- scenario and feature-schema metadata.

Every episode is strictly replay-validated. A trace with an infeasible action, mismatched replay metric, wrong feature schema, or incomplete 168 h forecast is rejected rather than silently used.

All three agents train from the same cached demonstrations. The current-state variant discards the forecast at model input; the flat and TCN variants consume it. This keeps teacher actions and demonstration states identical across variants.

## BC and Kickstarting

The existing imitation stack must be generalized from a single NumPy observation array to structured observations. Collection, mini-batching, `evaluate_actions`, sample weighting, and the kickstarting callback must all support current-state-only, flat-forecast, and TCN-forecast inputs.

Non-WAIT vessel actions retain dimension-specific up-weighting. Forecast encoder parameters participate in BC, PPO, and kickstarting updates. No separate autoencoder pretraining is introduced.

The kickstarting coefficient, decay schedule, BC episodes, BC epochs, action weights, PPO timesteps, PPO hyperparameters, and model seeds are identical across variants.

## Training and Evaluation Protocol

### Stage 1: local and short-queue smoke tests

- One 24 h demonstration episode with a full 168 h forecast.
- One BC epoch for each observation variant.
- One short PPO rollout for each variant.
- Verify CUDA placement, finite losses, action-mask compatibility, saved-model reload, and deterministic replay.

### Stage 2: pilot comparison

- One training seed.
- 30 MPC demonstration episodes.
- 100,000 PPO timesteps per variant.
- Five held-out paired evaluation seeds.

### Stage 3: formal comparison

- Three independent model seeds.
- The same fixed demonstration cache for every model seed.
- Ten held-out paired environment seeds per model.
- Report per-seed results, mean, standard deviation, 95% confidence interval, and paired differences.

Training seeds, demonstration seeds, and evaluation seeds are disjoint. The exact lists are recorded in the run manifest.

### Reference controllers

Evaluation includes idle, greedy, and `RollingNativeMpcController` references on the same held-out disturbance trajectories. For each learned variant, both BC-only and final PPO checkpoints are evaluated, with stochastic and deterministic policies reported separately.

## Metrics

Primary metric:

- vented tonnes over 720 h.

Secondary metrics:

- loss rate;
- end-of-episode unstored inventory, split by emitter, vessel, and terminal;
- stored tonnes and storage rate;
- operating cost and actual total cost;
- cost per stored tonne;
- longest venting streak;
- berth waiting, well throttling, and pressure-risk hours;
- MPC imitation negative log-likelihood, per-action accuracy, joint-action accuracy, and non-WAIT vessel-action accuracy;
- wall-clock demonstration, BC, and PPO training time;
- policy parameter count and inference latency.

The main comparison uses paired deltas on identical evaluation seeds. A variant is not declared better solely from its best seed.

## Required Ablations

The three requested variants are the required ablation. No reward or teacher changes are mixed into this comparison.

If the TCN variant wins, a later ablation may vary its latent width. If all forecast variants accumulate excessive terminal inventory, downstream-progress potential shaping becomes a separate reward experiment rather than a post-hoc change to this run.

## HPC Layout

MPC demonstration generation is CPU-heavy and runs separately from GPU PPO training. The workflow is:

1. environment and dependency smoke job;
2. replay-validated demonstration-generation job or seed-indexed job array;
3. cache audit and manifest creation;
4. three GPU pilot jobs using the same cache;
5. formal model-seed job array only after pilot validation;
6. paired evaluation and aggregate report job.

Formal GPU training uses Borg `root` partition with `long` QoS, one GPU per training job, checkpointing, unbuffered logs, and explicit output directories. Jobs record git commit, environment, configuration, seed sets, observation schema hash, and demonstration-cache hash.

## Failure Handling

- Reject forecast arrays with wrong shape, non-finite values, or channel-order mismatch.
- Reject a demonstration cache whose scenario or normalization metadata differs from training.
- Fail on MPC replay mismatch; do not fall back to greedy demonstrations.
- Keep the three variants' seeds and hyperparameters locked through a shared run manifest.
- Save intermediate demonstration shards and PPO checkpoints so SLURM timeouts can resume safely.
- Keep user-owned worktree changes outside the experiment commit scope.

## Verification and Success Criteria

Implementation is ready for a pilot only when:

1. current-state, flat-forecast, and TCN observations have tested shapes and deterministic feature order;
2. the same sampled seed gives identical current state and forecast to MPC and RL;
3. forecasts remain 168 h long at RL hour 719;
4. cached MPC actions replay exactly and are legal under the stored masks;
5. BC and kickstarting update the correct extractor parameters for all variants;
6. a short MaskablePPO run completes, saves, reloads, and evaluates for each variant;
7. the three variants differ only in forecast exposure/encoding;
8. the pilot report contains paired venting, inventory, storage, and cost results plus imitation and runtime metrics.

The experiment answers two separate questions: whether access to the 168 h forecast improves control, and whether a temporal encoder uses that same forecast more effectively than a flat MLP.
