# GNN attribution 3x2 BC results

All methods use forecast schema v4, decision-only BC, model seeds 0--4, and paired evaluation seeds 101--120. Values are mean +/- sample SD across the five model-seed means.

## Deterministic

| Method | Vented t | Stored t | Operating EUR | Total EUR | Operating EUR/t | Total EUR/t |
|---|---:|---:|---:|---:|---:|---:|
| Small MLP + Original TCN | 2,940.891 +/- 479.592 | 107,051.877 +/- 954.047 | 1,597,280.964 +/- 10,424.705 | 1,832,552.208 +/- 28,620.783 | 14.937 +/- 0.163 | 17.191 +/- 0.405 |
| Small MLP + FixedScale TCN | 3,867.230 +/- 1,217.570 | 105,394.870 +/- 1,646.976 | 1,597,415.100 +/- 26,229.093 | 1,906,793.507 +/- 74,946.310 | 15.176 +/- 0.183 | 18.237 +/- 1.077 |
| Large MLP + Original TCN | 5,270.376 +/- 1,251.244 | 103,130.480 +/- 1,842.623 | 1,579,647.157 +/- 19,805.051 | 2,001,277.274 +/- 89,903.843 | 15.344 +/- 0.216 | 19.598 +/- 1.285 |
| Large MLP + FixedScale TCN | 6,027.601 +/- 2,047.482 | 101,952.005 +/- 2,724.127 | 1,576,084.185 +/- 29,458.533 | 2,058,292.241 +/- 137,616.741 | 15.487 +/- 0.191 | 20.379 +/- 1.997 |
| Edge-GNN + Original TCN | 9,060.016 +/- 4,193.617 | 96,500.460 +/- 5,338.569 | 1,552,074.512 +/- 41,203.217 | 2,276,875.822 +/- 299,566.423 | 16.168 +/- 0.562 | 24.221 +/- 4.869 |
| Edge-GNN + FixedScale TCN | 6,889.766 +/- 2,250.367 | 99,592.007 +/- 2,000.207 | 1,579,898.259 +/- 16,088.925 | 2,131,079.517 +/- 168,963.032 | 15.912 +/- 0.203 | 21.665 +/- 2.266 |

## Stochastic

| Method | Vented t | Stored t | Operating EUR | Total EUR | Operating EUR/t | Total EUR/t |
|---|---:|---:|---:|---:|---:|---:|
| Small MLP + Original TCN | 12,671.515 +/- 2,616.904 | 93,751.153 +/- 3,380.477 | 1,564,604.089 +/- 10,470.592 | 2,578,325.310 +/- 202,918.607 | 16.759 +/- 0.553 | 28.003 +/- 3.191 |
| Small MLP + FixedScale TCN | 8,690.691 +/- 3,417.116 | 98,768.492 +/- 5,097.878 | 1,584,073.214 +/- 13,278.741 | 2,279,328.511 +/- 263,067.431 | 16.113 +/- 0.756 | 23.512 +/- 4.025 |
| Large MLP + Original TCN | 7,940.531 +/- 1,649.098 | 99,805.565 +/- 2,415.550 | 1,581,604.338 +/- 18,865.875 | 2,216,846.786 +/- 118,999.883 | 15.884 +/- 0.311 | 22.576 +/- 1.725 |
| Large MLP + FixedScale TCN | 8,890.117 +/- 1,494.342 | 98,643.294 +/- 1,654.242 | 1,576,331.469 +/- 20,237.511 | 2,287,540.843 +/- 115,571.735 | 16.017 +/- 0.227 | 23.416 +/- 1.506 |
| Edge-GNN + Original TCN | 19,036.695 +/- 4,195.493 | 85,166.163 +/- 4,866.829 | 1,520,664.124 +/- 29,430.302 | 3,043,599.748 +/- 307,056.975 | 17.984 +/- 0.786 | 36.827 +/- 6.627 |
| Edge-GNN + FixedScale TCN | 20,673.943 +/- 5,071.316 | 83,084.169 +/- 5,651.371 | 1,510,156.540 +/- 36,438.409 | 3,164,071.963 +/- 370,021.849 | 18.323 +/- 0.827 | 39.358 +/- 7.220 |

## Primary paired deterministic contrasts

Differences are left minus right; intervals are two-sided 95% t intervals over five paired model-seed means.

| Contrast | Metric | Mean difference | 95% CI |
|---|---|---:|---:|
| future_effect_small | vented_t | 926.340 | [-429.875, 2,282.554] |
| future_effect_large | vented_t | 757.224 | [-1,895.962, 3,410.410] |
| future_effect_edge | vented_t | -2,170.251 | [-9,348.024, 5,007.523] |
| graph_effect_original | vented_t | 3,789.640 | [-2,508.587, 10,087.867] |
| graph_effect_fixed | vented_t | 862.165 | [-2,842.094, 4,566.425] |
| graph_by_future_interaction | vented_t | -2,927.475 | [-11,446.435, 5,591.486] |
| future_effect_small | stored_t | -1,657.007 | [-3,862.189, 548.175] |
| future_effect_large | stored_t | -1,178.476 | [-4,907.814, 2,550.862] |
| future_effect_edge | stored_t | 3,091.547 | [-4,882.902, 11,065.995] |
| graph_effect_original | stored_t | -6,630.020 | [-14,556.775, 1,296.735] |
| graph_effect_fixed | stored_t | -2,359.998 | [-7,214.930, 2,494.935] |
| graph_by_future_interaction | stored_t | 4,270.022 | [-5,421.128, 13,961.172] |
| future_effect_small | total_cost | 74,241.299 | [-14,423.191, 162,905.789] |
| future_effect_large | total_cost | 57,014.967 | [-105,107.819, 219,137.752] |
| future_effect_edge | total_cost | -145,796.305 | [-673,113.547, 381,520.938] |
| graph_effect_original | total_cost | 275,598.548 | [-172,821.377, 724,018.473] |
| graph_effect_fixed | total_cost | 72,787.277 | [-185,398.764, 330,973.317] |
| graph_by_future_interaction | total_cost | -202,811.271 | [-824,449.068, 418,826.525] |
| future_effect_small | total_cost_per_stored_t | 1.046 | [-0.285, 2.377] |
| future_effect_large | total_cost_per_stored_t | 0.782 | [-1.709, 3.273] |
| future_effect_edge | total_cost_per_stored_t | -2.556 | [-10.447, 5.335] |
| graph_effect_original | total_cost_per_stored_t | 4.623 | [-2.441, 11.688] |
| graph_effect_fixed | total_cost_per_stored_t | 1.286 | [-2.553, 5.124] |
| graph_by_future_interaction | total_cost_per_stored_t | -3.338 | [-12.514, 5.838] |

## Forecast-use diagnostics

| Method | Active seeds | Feature L2 | Input gradient L2 | Shuffle TV | Argmax change |
|---|---:|---:|---:|---:|---:|
| Small MLP + Original TCN | 1/5 | 0.483 | 1.196e-04 | 0.0057 | 1.21% |
| Small MLP + FixedScale TCN | 5/5 | 3.923 | 1.164e-03 | 0.0295 | 6.62% |
| Large MLP + Original TCN | 2/5 | 0.775 | 2.549e-04 | 0.0094 | 2.28% |
| Large MLP + FixedScale TCN | 5/5 | 3.902 | 6.584e-04 | 0.0163 | 4.07% |
| Edge-GNN + Original TCN | 1/5 | 1.580 | 1.056e-03 | 0.0033 | 0.45% |
| Edge-GNN + FixedScale TCN | 5/5 | 2.400 | 5.474e-05 | 0.0028 | 0.40% |
