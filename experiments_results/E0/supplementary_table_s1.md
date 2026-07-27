# Supplementary Table S1 — E0 physical validation

| Check | Category | Expected | Observed | Tolerance | Status |
|---|---|---|---:|---|---:|
| mass.formal_720h | Mass conservation | Maximum absolute whole-system residual is near zero | 6.158370524644852e-08 | ≤ 1e-06 t | **PASS** |
| capacity.emitter | Capacity limits | emitter inventory does not exceed capacity | 1.0 | ≤ 1 + 1e-06 t-equivalent | **PASS** |
| capacity.vessel | Capacity limits | vessel inventory does not exceed capacity | 1.0 | ≤ 1 + 1e-06 t-equivalent | **PASS** |
| capacity.terminal | Capacity limits | terminal inventory does not exceed capacity | 1.0 | ≤ 1 + 1e-06 t-equivalent | **PASS** |
| capacity.reservoir | Capacity limits | reservoir inventory does not exceed capacity | 0.04533188758111156 | ≤ 1 + 1e-06 t-equivalent | **PASS** |
| capacity.nonnegative_inventory | Capacity limits | All entity inventories remain non-negative | 0.0 | ≥ −1e-06 t | **PASS** |
| pressure.reservoir | Pressure limits | Reservoir pressure does not exceed its limit | 0.0 | ≤ 1e-08 bar | **PASS** |
| injection.maximum_feasible | Automatic well control | Actual injection never exceeds the requested continuous maximum | 0.0 | ≤ 1e-06 t/h | **PASS** |
| injection.common_rule | Automatic well control | All controller labels call the same state-to-rate rule | 0.0 | 0 t/h | **PASS** |
| state.destination_lock | Vessel state machine | No destination changes while a vessel remains mid-voyage | 0 | 0 events | **PASS** |
| state.terminal_fifo | Vessel state machine | Terminal queues never contain duplicate vessels | 0 | 0 hours | **PASS** |
| state.hard_violations | Physical feasibility | No hard physical violations in the 720 h rollout | 0 | 0 | **PASS** |
| cost.decomposition | Economic accounting | Ledger components sum to total episode cost | 0.0 | ≤ 1e-6 EUR | **PASS** |
| scenario.no_disturbance | Simplified scenarios | No weather, maintenance, or high-output event is present | speed=1.0; maintenance=0; capture_max=1.0 | speed=1; maintenance=0; capture_max=1 | **PASS** |
| scenario.single_vessel | Simplified scenarios | One-emitter/one-vessel case completes without hard violations | 0 | 0 | **PASS** |
| scenario.weather | Simplified scenarios | Weather window reduces the vessel speed factor | 0.6 | < 1 | **PASS** |
| scenario.maintenance | Simplified scenarios | Maintenance occurs and blocks injection | samples=168; max_unavailable_injection=0.0 | samples > 0 and injection ≤ 1e-6 t/h | **PASS** |
| scenario.high_output | Simplified scenarios | High-output multiplier exceeds the nominal capture rate | 1.5 | > 1 | **PASS** |
| cleanup.current_state | Terminal accounting | Common compact cleanup is deterministic and non-negative | 0.0 | repeat error ≤ 1e-6 EUR | **PASS** |
| tests.e0_suite | Automated regression | All selected physical, disturbance, state, cost, replay, and cleanup tests pass | 179/179 passed | 0 failures and 0 errors | **PASS** |
