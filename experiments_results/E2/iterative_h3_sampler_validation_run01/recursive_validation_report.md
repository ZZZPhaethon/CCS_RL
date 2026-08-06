# Iterative H3 recursive validation

Validation-only P2–P4 comparison; no formal test stage is included.

- `b_gate_only` P3 minus P2: +46.8 kEUR/episode (95% hierarchical CI -5.6 to +113.1).
- `b_gate_only` P4 minus P2: -0.3 kEUR/episode (95% hierarchical CI -54.4 to +64.2).
- `c_dedup_balanced` P3 minus P2: -11.8 kEUR/episode (95% hierarchical CI -101.9 to +60.5).
- `c_dedup_balanced` P4 minus P2: -31.4 kEUR/episode (95% hierarchical CI -129.5 to +75.9).

## Fixed validation references

- `b_gate_only` P4 minus `one_shot_matched`: -37.5 kEUR/episode (95% hierarchical CI -111.7 to +49.9).
- `b_gate_only` P4 minus `iterative_p4`: -9.5 kEUR/episode (95% hierarchical CI -70.3 to +33.5).
- `c_dedup_balanced` P4 minus `one_shot_matched`: -44.2 kEUR/episode (95% hierarchical CI -112.0 to +21.4).
- `c_dedup_balanced` P4 minus `iterative_p4`: -16.1 kEUR/episode (95% hierarchical CI -92.5 to +49.7).

## Retained-variant comparison

- `c_dedup_balanced` P4 minus `b_gate_only` P4: -6.7 kEUR/episode (95% hierarchical CI -80.2 to +57.2).
