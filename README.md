<h1 align="center">
  <img src="assets/Logo.png" alt="Logo" width="400"/>
</h1>
<h3 align="center">
Physics-Constrained Simulation and Learned Dispatch Control for Ship-Based CCS Chains
</h3>
<p align="center">
  Languages: English | <a href="README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="MIT"></a>
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue?style=flat-square" alt="Python >=3.10">
  <a href="https://drive.google.com/drive/folders/147lfZ1M1d3Am0v65fk1SX0jsXmk2lVzN"><img src="https://img.shields.io/badge/Google%20Drive-Dataset-4285F4?logo=googledrive&logoColor=white&style=flat-square" alt="Google Drive Dataset"></a>
</p>

<p align="center">
  <img src="assets/CCUSoverview.png" alt="CCUS teaser" width="95%">
</p>

CCS_RLLLM is a research codebase for **hourly operational dispatch of ship-based CO₂
transport and storage**. It combines a physics-constrained simulator of the full CCS chain
with a family of controllers — heuristics, rolling MILP, PPO variants, and the current main
method **Iterative Action-Q** — evaluated against each other under one frozen disturbance and
information protocol.

The physical chain is:

```text
Emitter -> Vessel -> Terminal -> Pipeline -> SubseaManifold -> InjectionWell -> Reservoir
```

Controllers only decide **vessel dispatch**. Injection is handled by a shared automatic
well controller that always takes the highest feasible rate, so no method gains an advantage
by relaxing physics. `src/sim/` is the single source of physical truth: it validates every
action, advances capture, sailing, loading, unloading, injection and reservoir pressure, and
returns auditable trajectories, cost ledgers and KPIs.

---

## Research question

> In a three-vessel CCS transport–storage system with weather, capture and injection-capacity
> disturbances, can a method that keeps Greedy as its safe default and learns a **small number
> of high-value interventions** from counterfactual rollouts reduce total cost and CO₂ venting
> at low online decision cost?

This is the hypothesis under test, not a settled result. The full argument structure lives in
[`docs/paper_structure_zh.md`](docs/paper_structure_zh.md); the experiment design is locked in
[`docs/paper_experiment_plan_zh.md`](docs/paper_experiment_plan_zh.md).

## Method — Iterative Action-Q

Iterative Action-Q is neither PPO nor plain behaviour cloning. Greedy provides the safe default
action and the first state distribution; the model then repeatedly collects the states **it
would itself visit**:

```text
Greedy states G0        -> train P1
P1 roll-in states G1    -> train P2 on G0+G1
P2 roll-in states G2    -> train P3 on G0+G1+G2
P3 roll-in states G3    -> train P4 on all data
```

- **Label.** Every candidate action starts from the same decision state and is simulated to the
  end of the 720 h episode. The target is `1e-5 x (baseline total cost - candidate total cost)`,
  so a positive target means the candidate genuinely saves money over the whole horizon — not a
  one-hour reward and not a truncated 168 h return.
- **Action space.** Each vessel picks one of `WAIT / Terminal / 3 emitters / FOLLOW`; three
  vessels give up to `6³ = 216` joint actions, with infeasible ones masked out.
- **Network.** Shared vessel encoder, structured action embeddings, 5 bootstrap heads, 51
  quantiles per action.
- **Deployment gate.** The policy overrides Greedy only under agreement and margin thresholds
  (e.g. ≥4/5 heads agree, predicted gain above ≈€40k), with a capped number of overrides split
  across fixed intervention windows. Otherwise it executes `FOLLOW`.

Details and the exact production configuration: [`docs/iterative_action_q_training_zh.md`](docs/iterative_action_q_training_zh.md).

## Unified comparison protocol — `unified_window_v1`

All controllers share the same three-vessel network, 720 h episodes, 1 h physical step, initial
state sampling, economic parameters, action masks and per-seed disturbance trajectories. The
protocol is machine-readable and frozen:

- [`experiments/protocols/unified_window_v1_paper_protocol.json`](experiments/protocols/unified_window_v1_paper_protocol.json) — environment, disturbance rates, well rule, forecast protocol, cost formula, training budget.
- [`experiments/protocols/unified_window_v1_seed_manifest.json`](experiments/protocols/unified_window_v1_seed_manifest.json) — train / validation / test seed ranges.

| Controller | Paper name | Trained | Runtime future information |
|---|---|---|---|
| Fixed vessel–emitter split | Fixed-Assignment Heuristic | no | none |
| Dynamic greedy shuttle | Greedy | no | none |
| Direct one-action-per-hour PPO | Hourly Centralized Maskable PPO | yes | shared 168 h summary |
| Event-based v4 architecture | Event-Residual PPO | yes | shared 168 h summary |
| **Current main method** | **Iterative Action-Q** | yes | shared 168 h summary |
| Rolling optimisation | Rolling MILP | no | full hourly 168 h forecast |
| Offline reference only | Full-horizon MILP (time-limited) | no | perfect foresight |

Key fairness rules: identical forecast **source** for every forecast-capable method (currently a
perfect-forecast protocol), a matched environment-interaction budget
`B_selected = 9,505,319` simulator-hour calls for the three
learning methods, a common compact trip-cleanup terminal value applied to every method's reported
cost, and unvisited paired-comparison test seeds `9,000,031–9,000,060`. This range is locked
against further selection; `9,000,001–9,000,030` is retained as a deprecated historical test block.

## Preliminary results

Development-seed comparison (`8,000,001–8,000,030`, 30 paired seeds, **one training seed per
learning method**). These remain development-only results.
Full write-up: [`docs/preliminary results/unified_window_control_comparison_2026-07-26_zh.md`](docs/preliminary%20results/unified_window_control_comparison_2026-07-26_zh.md).

| Method | Total cost (EUR) | vs Greedy | Vented (t) | Stored (t) | EUR/t | Wins vs Greedy |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 2,059,907 | — | 7,883.1 | 100,959.4 | 21.01 | — |
| Residual PPO v4 | 1,942,032 | −117,876 | 5,263.1 | 103,421.5 | 18.87 | 13/30 |
| Iterative Q (state only) | 1,699,864 | −360,043 | 1,704.5 | 108,989.6 | 15.68 | 23/30 |
| **Iterative Q (24/72 h future)** | **1,633,631** | **−426,276** | **821.1** | **109,242.1** | **14.97** | **25/30** |
| Hybrid RL (Greedy base) | 3,134,909 | +1,075,002 | 22,990.4 | 87,728.0 | 36.05 | 2/30 |

Paired 95% bootstrap CIs exclude zero for both Iterative Q variants but **not** for Residual PPO
v4 ([−309,118, +47,944]), so v4 cannot yet be called reliably better than Greedy. Adding the
24/72 h future summary buys a further −66,233 EUR on average, CI [−119,717, −15,833].

## Installation

Run everything from the repository root.

```powershell
uv sync
uv run python -m pip install -e .
```

With RL dependencies (`numpy`, `gymnasium`, `stable-baselines3`, `sb3-contrib`):

```powershell
uv sync --extra rl
uv run python -m pip install -e ".[rl]"
```

Without `uv`:

```powershell
pip install -e ".[rl]"
```

Additional requirements:

- Python `>=3.10` (GPU training environment uses 3.12).
- Core physical layer: `searoute>=1.6`, `CoolProp>=6.6`.
- Iterative Action-Q and the event-based stack additionally need `torch`.
- Rolling MILP / Full-horizon MILP need a CPLEX installation; CBC is far too slow for 720 h
  multi-seed studies.
- Wave-height prediction training uses a separate conda environment:

```powershell
conda env create -f environment-gpu.yml
conda activate ccs-rlllm-gpu
pip install -e ".[rl]"
```

## Quick start

### Physical-layer demo and dashboards

```powershell
uv run python examples\run_physical_layer_demo.py
uv run python examples\build_phase1_dashboard.py
uv run python examples\build_rule_based_dashboards.py
```

### Iterative Action-Q pipeline

The four stages map directly onto CLI entry points. On a cluster, use the launcher instead
(next section) — it wires the whole dependency chain.

```powershell
# 1. G0 - Greedy roots
uv run python -m experiments.generate_iterative_q_greedy_data `
  --out-path output\iq\g0_train.pt --split train `
  --seeds (1500..1699) --roots-per-seed 12

# 2. Train P1 from scratch on G0
uv run python scripts\train_iterative_action_q.py `
  --train-data output\iq\g0_train.pt --validation-data output\iq\g0_val.pt `
  --out-dir output\iq\p1 --observation-input v4_future_24_72

# 3. G1-G3 - roll the current policy in, keep the states it visits
uv run python -m experiments.create_iterative_q_lock `
  --checkpoint output\iq\p1\iterative_action_q.pt --out-path output\iq\p1\lock.json `
  --protocol-id unified_window_v1 --residual-margin 0.40 --economic-margin-eur 40000
uv run python -m experiments.generate_iterative_q_policy_data `
  --lock-config output\iq\p1\lock.json --out-path output\iq\g1_train.pt `
  --split train --seeds (1500..1539)

# 4. Retrain on cumulative data, then evaluate against Greedy on unseen seeds
uv run python scripts\train_iterative_action_q.py `
  --train-data output\iq\g0_train.pt output\iq\g1_train.pt `
  --validation-data output\iq\g0_val.pt --initial-checkpoint output\iq\p1\iterative_action_q.pt `
  --out-dir output\iq\p2 --observation-input v4_future_24_72
uv run python -m experiments.evaluate_iterative_action_q `
  --checkpoint output\iq\p4\iterative_action_q.pt --out-dir output\iq\p4\eval
```

`--observation-input` selects the ablation arm: `state_only`, `v4_future_24_72` (the locked E1
representation), `forecast_168`, or one of the summary window/band variants. `--forecast-encoder`
switches between `small_mlp`, `tcn` and `gru`.

### Unified controller comparison

```powershell
uv run python -m experiments.compare_unified_window_controls `
  --iterative-q-checkpoint output\iq\p4\iterative_action_q.pt `
  --v4-run-dir output\event_v4\run1 `
  --out-dir output\comparison
```

Writes per-seed CSV and a summary JSON with cost components, vent/stored tonnes, override counts
and wall-clock time.

### Cluster runs

`hpc/launch_iterative_action_q.sh` chains data generation, staged training and evaluation into one
SLURM dependency graph. Check the configuration without submitting anything:

```bash
DRY_RUN=1 bash hpc/launch_iterative_action_q.sh
```

Override `PROJECT_DIR`, `RUN_ROOT` and `CONFIG_NAME` for your own paths and run directory; the
launcher refuses to overwrite an existing run directory and writes all job IDs to
`RUN_ROOT/job_ids.txt`. Individual `hpc/submit_*.sh` scripts cover single stages (data, training,
evaluation, ablations, environment checks).

### Other controllers

```powershell
# Formal hourly centralized Maskable PPO (no BC, event trigger, or executor)
uv run python -m sim.control.hourly_ppo.train_hourly_ppo `
  --episode-hours 720 --forecast-context-hours 168 `
  --future-summary-windows-h 168 --gamma 1 `
  --max-simulator-hour-steps 9505319

# Forecast-encoder comparison (demos -> merge -> train -> report subcommands)
uv run python scripts\compare_forecast_encoders_rl.py --help

# Wave-height prediction models
uv run python -m sim.scenario_generation.wave_height.prediction.train_lstm
```

## Tests

```powershell
uv run python -m unittest discover -s tests
```

Without an editable install:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m unittest discover -s tests
```

## Repository layout

```text
CCS_RLLLM/
|-- data/                 # Capture-rate profiles and external reference data
|-- docs/                 # Paper plan, method notes, and dated preliminary results
|-- examples/             # Physical-layer demos and dashboard builders
|-- experiments/          # Data generation, evaluation, comparison, ablation analysis
|   `-- protocols/        # Frozen paper protocol and seed manifest
|-- hpc/                  # SLURM launchers and per-stage submission scripts
|-- scenarios/            # Reproducible scenario JSON files
|-- scripts/              # Training entry points
|-- src/sim/              # Main Python package
|-- tests/                # 58 unit, structure, and experiment smoke tests
|-- environment-gpu.yml   # GPU training environment
`-- pyproject.toml
```

### `src/sim` structure

```text
src/sim/
|-- actions/              # ActionProposal, ActionFrame, ActionResolver
|-- control/              # Controllers - see below
|-- entities/             # Emitter, vessel, terminal, pipeline, well, reservoir state
|-- environment/          # CCSEnv, factories, forecast/past observations, Gym adapters
|-- operations/           # Capture, loading, unloading, transport, injection, pressure limits
|-- scenario_generation/  # Disturbance and wave-height scenario generation
|-- visualization/        # Dashboard payloads and HTML rendering
|-- economics.py          # Cost and revenue model
|-- line_source.py        # Reservoir/well pressure line-source model
|-- metrics.py            # Rollouts, KPIs, evaluation summaries
|-- network.py            # Physical network graph and single-step settlement
|-- network_scenarios.py  # Build Northern Lights networks from JSON/data
|-- routes.py             # Route and distance calculation
|-- ship_speed.py         # Sea-state effects on vessel speed
`-- simulator.py          # High-level simulation runner
```

### `src/sim/control` — controller families

```text
control/
|-- baselines.py                # Idle and greedy shuttle policies
|-- rule_based.py               # Fixed-assignment and rule controllers
|-- milp.py / cplex_milp.py     # Static MILP benchmark and CPLEX backend
|-- rolling_milp.py             # Rolling-horizon MILP with replay-validated warm start
|-- shikha2025.py               # Paper Lagrangean + shrinking-horizon reproduction
|-- native_mpc.py               # Multi-candidate native MPC
|-- iterative_action_q.py       # Main method: the production Q network
|-- hourly_ppo/                 # Direct one-policy-action-per-hour PPO baseline
|-- recurrent_distributional_q.py
|-- imitation.py / demonstrations.py / replay.py
`-- event_based/                # Algorithm layer, outside the physics layer
    |-- contracts.py            # DispatchGoal boundary: high-level policy <-> executor
    |-- evaluation.py           # Physical rollout evaluator for fair comparison
    |-- hybrid/                 # Rule, native-MPC and rolling-MILP executors
    |-- rl/                     # Sparse 24 h high-level PPO
    `-- residual_rl{,_v2,_v3,_v4}/  # Residual intervention PPO, v4 is Event-Residual PPO
```

`event_based/` decides *which operating goal to pursue*; it must never add capacities, pressure
equations or clipping rules. Those stay in `entities/`, `operations/` and `network.py` so every
controller receives identical physics. See [`src/sim/control/event_based/README.md`](src/sim/control/event_based/README.md).

## Documentation map

| Document | Contents |
|---|---|
| [`docs/paper_structure_zh.md`](docs/paper_structure_zh.md) | Paper argument chain, section-by-section evidence requirements |
| [`docs/paper_experiment_plan_zh.md`](docs/paper_experiment_plan_zh.md) | Formal E0/E1/E6/E4/E7/E3/E5 experiment sequence, fairness protocol, metrics, statistics |
| [`docs/iterative_action_q_training_zh.md`](docs/iterative_action_q_training_zh.md) | Method definition, production configuration, code entry points |
| [`docs/shikha2025_reproduction_zh.md`](docs/shikha2025_reproduction_zh.md) | Shikha et al. (2025) Lagrangean + shrinking-horizon reproduction on the shared case |
| [`docs/preliminary results/`](docs/preliminary%20results/) | Dated result records — controller comparison, encoder comparison, future-adapter and reproducibility ablations |
| [`docs/CCS_RL_Research_Core_Idea.md`](docs/CCS_RL_Research_Core_Idea.md) | Original research framing |
| [`docs/physical_layer_v1_cn.md`](docs/physical_layer_v1_cn.md) | Physical-layer model specification |
| [`docs/northern_lights_line_source_pressure_study.md`](docs/northern_lights_line_source_pressure_study.md) | Reservoir pressure line-source study |
| [`docs/experiments_summary.md`](docs/experiments_summary.md) | Historical record of the earlier RL/LLM phase (scripts since removed) |
| `src/sim/scenario_generation/wave_height/prediction/README.md` | Wave-height prediction models |

## Data

Large external files are not fully tracked in git. Download them and restore them into the
matching directories before running the related scripts.

- Google Drive: <https://drive.google.com/drive/folders/147lfZ1M1d3Am0v65fk1SX0jsXmk2lVzN?usp=sharing>
- `scenarios/` — scenario JSON, including `northern_lights_phase1_3vessels.json`, the network used
  by the paper protocol.
- `data/capture_rates/` — Phase 1/Phase 1+ emitter capture-rate profiles and metadata.
- `data/网络收集资料/` — curated external references such as Climate TRACE source mapping.

## Extending the codebase

**Add a controller.** Implement it under `src/sim/control/` (algorithm-layer controllers go in
`event_based/`), express actions as `ActionProposal` / `ActionFrame`, route them through
`ActionResolver` into `network.step()`, register it in the comparison experiment, and add a
behaviour test under `tests/`.

**Add a scenario.** Add the JSON under `scenarios/`, put any new capture profile in
`data/capture_rates/`, add a loading entry point in `src/sim/network_scenarios.py` or an
environment factory, and validate it with a demo or comparison run.

**Add a disturbance.** Generate the episode time series in
`src/sim/scenario_generation/generator.py`, define runtime resolution in
`disturbance_resolver.py`, connect it to `CCSEnv` or the relevant operation module, and add
fixed-seed tests. Changing any `unified_window_v1` disturbance parameter requires a **new protocol
version** and a rerun of all methods.

## Roadmap

- [x] Physical entities, operation modules, network step, and pressure limits.
- [x] Action proposal/resolver protocol layer.
- [x] Rule-based, static MILP, rolling MILP and native MPC controllers.
- [x] Gymnasium/SB3 RL environments and PPO/BC training entry points.
- [x] Event-based algorithm layer with hybrid executors and residual RL v1–v4.
- [x] Iterative Action-Q training, evaluation and gating.
- [x] Frozen `unified_window_v1` protocol and seed manifest.
- [ ] Implement the pending protocol requirements: shared automatic well rule in every controller
      interface, complete cost/activity diagnostics, and the 1 h simulator-step counter for `B_4800`.
- [ ] Retrain Hourly Centralized Maskable PPO and Event-Residual PPO with objective-aligned rewards.
- [ ] Run ≥3 independent training seeds and report future frozen-controller comparisons on
      unvisited test set `9,000,031–9,000,060`.
- [ ] Replace personal paths in HPC scripts with environment-variable configuration.
- [ ] Package large datasets and model weights as downloadable release assets.

## 📝 Citation

```bibtex
@software{ccs_rlllm,
  title  = {CCS_RLLLM: Physics-Constrained Simulation and Learned Dispatch Control for Ship-Based CCS},
  author = {CCS_RLLLM contributors},
  year   = {2026},
  note   = {Research code for CCS chain simulation, optimisation control, and reinforcement learning}
}
```

⭐ **If you find this work useful, please star the repository!**
