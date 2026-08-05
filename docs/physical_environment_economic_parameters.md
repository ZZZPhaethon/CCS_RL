# Physical-Environment Economic Parameters

| Type | Parameter | Model field | Value | Unit | Application in the physical environment |
|---|---|---|---:|---|---|
| Input | Currency | `currency` | EUR | N/A | Currency used for all costs and penalties. |
| Input | Carbon price / venting penalty | `carbon_price_eur_per_t` | 80.00 | EUR/t CO2 vented | Applied to CO2 that is vented or otherwise lost. |
| Input | Ship-fuel price | `ship_fuel_cost_hfo_eur_per_t` | 600.00 | EUR/t fuel | Multiplied by calculated engine fuel consumption; the field name is retained for legacy compatibility. |
| Input | Main-engine specific fuel consumption | `main_engine_fuel_use_kg_per_kwh` | 0.148 | kg/kWh | Converts engine energy use into fuel mass. |
| Input | Main-engine rated power | `main_engine_power_kw` | 5,500 | kW | Reference power used for sailing and hoteling fuel calculations. |
| Input | Cruise power fraction | `cruise_power_fraction` | 0.85 | fraction of rated power | Applied while a vessel is sailing. |
| Input | Hoteling power fraction | `hoteling_power_fraction` | 0.05 | fraction of rated power | Applied during loading and unloading service time. |
| Input | Source-side CO2 conditioning cost | `conditioning_eur_per_t` | 7.82 | EUR/t CO2 loaded | Applied to CO2 conditioned before ship export. |
| Input | Terminal-side CO2 reconditioning cost | `reconditioning_eur_per_t` | 0.41 | EUR/t CO2 | Applied when CO2 is transferred from the terminal toward pipeline injection. |
| Input | Storage-shortfall penalty | `storage_shortfall_eur_per_t` | 0.00 | EUR/t shortfall | Disabled in the formal economic objective; storage shortfall is reported as a KPI. |
| Derived | Vessel sailing fuel cost | `vessel_fuel_eur_per_h_sailing` | 415.14 | EUR/vessel-h | Charged for each vessel-hour spent sailing. |
| Derived | Loading/unloading hoteling cost | `hoteling_fuel_eur_per_h` | 24.42 | EUR/service-h | Charged separately for loading time and unloading time. |

The model includes variable operating costs only. CAPEX, bundled transport-and-storage tariffs, fixed port-call or berth fees, separate waiting tariffs, and separate injection-electricity costs are excluded. Episode cost is the sum of vessel fuel, conditioning, reconditioning, loading, unloading, venting penalty, and storage-shortfall penalty; reported total cost additionally includes the common end-of-horizon cleanup operating cost.

Sources: `src/sim/economics.py` and `experiments/protocols/unified_window_v1_paper_protocol.json`.
