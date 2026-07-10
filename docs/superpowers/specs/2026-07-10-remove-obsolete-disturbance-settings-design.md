# Remove Obsolete Disturbance Settings

## Goal

Remove the obsolete load-shift disturbance controls and the configurable
injectivity-decline generator from active project code without changing the
current MPC/RL behavior or removing the generic well-capacity interface used by
the simulator and optimization controllers.

## Scope

### Load shift

- Remove the broken `load_shift` import and conditional generator selection
  from `scripts/train_residual_rl.py`.
- Remove the `--load-shift` CLI flag and its output/report tags.
- Keep historical experiment records that describe previously completed
  load-shift studies; they are results, not active settings.
- Keep the project-structure assertion that the deleted
  `src/sim/scenario_generation/load_shift.py` module must remain absent.

### Injectivity decline

- Remove `ScenarioConfig.injectivity_max_decline`, `injectivity_floor`,
  `injectivity_noise_std`, and `injectivity_warmstart_min`.
- Remove the random injectivity stream and `_injectivity_series` generator.
- Materialize each scenario's `injectivity_factor` as a nominal all-ones series
  so existing simulator, observation, replay, MILP, and MPC interfaces remain
  stable.
- Remove now-invalid zero-valued decline arguments from experiments and tests.
- Replace decline/warm-start tests with a structural regression test proving
  that decline configuration is no longer exposed and sampled scenarios remain
  nominal.
- Remove current README claims that injectivity decline is a supported
  disturbance. Historical research notes may retain it as a previously
  considered idea.

## Preserved behavior

- `PhysicalState.injectivity_factor`, `Scenario.injectivity_factor`, snapshots,
  observations, and solver capacity calculations remain available.
- A factor of `1.0` means the well uses its nominal capacity; explicit factors
  supplied by tests or external scenario data can still derate a well.
- Well maintenance remains a separate supported all-or-nothing disturbance.
- Current Native MPC runs are unchanged because they already disable decline
  and load shift.

## Verification

1. Add regression tests that fail while the obsolete `ScenarioConfig` fields
   and residual-RL flag/import still exist.
2. Remove the production settings and generator code until those tests pass.
3. Run focused scenario, environment, residual-script, MILP/MPC, and project
   structure tests.
4. Run the full test suite and an exact repository search confirming that no
   active code exposes load-shift or injectivity-decline settings.

## Non-goals

- Removing the generic injectivity-factor state/interface.
- Changing well maintenance, pressure, or nominal maximum-injection physics.
- Rewriting historical experiment summaries or prior research-idea documents.
- Changing controller objectives, scenario topology, or economic parameters.
