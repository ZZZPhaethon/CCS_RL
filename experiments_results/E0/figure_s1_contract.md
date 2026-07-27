# Figure S1 contract

- Core conclusion: The physical simulator conserves whole-system CO2 mass
  throughout a disturbed 720 h rollout while inventories and storage respond
  consistently to operational disturbances.
- Figure archetype: quantitative grid.
- Target/output: double-column supplementary figure; editable SVG and PDF,
  600 dpi TIFF, 300 dpi PNG preview.
- Backend: Python/matplotlib only.
- Final size: 183 mm × 120 mm.
- Panel a: cumulative captured, stored, and vented CO2.
- Panel b: recoverable inventory split across emitters, vessels, and terminal.
- Panel c: absolute whole-system mass-balance residual and tolerance.
- Panel d: capture, weather-speed, and well-availability disturbance states.
- Evidence hierarchy: panel c is the conservation evidence; panels a–b show
  physical stock/flow consistency; panel d anchors changes to disturbances.
- Statistics: one deterministic validation trajectory, seed 8100001; no
  inferential statistics.
- Source data: source_data/figure_s1_timeseries.csv.
- Image integrity: vector-native line art; no local image adjustment.
- Reviewer risk: this representative trajectory does not replace the complete
  component-test suite reported in Supplementary Table S1.
