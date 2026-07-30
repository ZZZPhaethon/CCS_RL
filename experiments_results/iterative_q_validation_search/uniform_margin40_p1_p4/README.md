# Iterative Q uniform-margin 0.40 validation experiment

Date: 2026-07-28

## Question

Test whether using the deployment margin consistently during every learned-policy
roll-in materially degrades Iterative Q when no separate low-margin exploration
bank is used.

## Controlled protocol

- Reused the exact baseline G0 data and P1 checkpoint because neither depends on
  the policy roll-in margin.
- Verified reused P1 SHA256:
  `0c1ad6d240c064cf482b7d405573c2766c61457ffb1333e062857bbd4a1c4098`.
- Generated G1, G2, and G3 using P1, P2, and P3 locks with:
  - residual margin: `0.40`
  - economic margin: `€40,000`
- Used no low-margin exploration bank.
- Kept the baseline data counts, scenario seeds, model seed 0, optimizer,
  architecture, policy windows, maximum overrides, terminal cleanup, and
  20 validation seeds unchanged.
- Evaluated P2, P3, and P4 with the same exact-loop gate at margin `0.40`.

The original baseline used P1 roll-in margin `0.10`, later roll-in margins
`0.15`, and evaluation margin `0.40`.

## SLURM provenance

- Environment check: `33049`
- Main dependency chain: `33050`–`33060`
- P2 exact validation: `33066`
- P3 exact validation: `33067`
- P4 exact validation: `33060`
- All required jobs completed with exit code `0:0`.
- All recovered SLURM stderr logs are empty and contain no traceback markers.
- P1, P2, and P3 checkpoint hashes match their lock files.
- All P2/P3/P4 evaluation rows satisfy the cleanup and paired-cost identities
  exactly.

## Twenty-seed validation results

| Stage | Protocol | Mean total cost € | Mean delta vs Greedy € | Wins | Vent t | Stored t | Unit cost €/t |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 | shared checkpoint | 1,964,742.54 | -109,265.82 | 11/20 | 2,145.36 | 106,843.45 | 18.481 |
| P2 | original mixed margins | 1,887,957.31 | -186,051.05 | 17/20 | 1,371.55 | 109,027.07 | 17.351 |
| P2 | uniform 0.40 | 1,898,054.80 | -175,953.57 | 15/20 | 1,327.69 | 109,305.57 | 17.372 |
| P3 | original mixed margins | 1,837,041.35 | -236,967.01 | 17/20 | 615.78 | 109,873.76 | 16.774 |
| P3 | uniform 0.40 | 1,853,089.54 | -220,918.82 | 16/20 | 751.00 | 109,326.81 | 17.007 |
| P4 | original mixed margins | 1,865,713.25 | -208,295.12 | 17/20 | 1,151.38 | 109,137.82 | 17.108 |
| P4 | uniform 0.40 | **1,825,687.98** | **-248,320.38** | **17/20** | **728.10** | **109,924.88** | **16.624** |

Uniform-margin changes relative to the matching original stage:

- P2: `+€10,097.49` (`+0.535%`)
- P3: `+€16,048.19` (`+0.874%`)
- P4: `-€40,025.27` (`-2.145%`)

The uniform P4 is also `€11,353.37` cheaper on average than the previously
selected original P3. The paired 20-seed confidence interval for this difference
still includes zero, so this is evidence against a large degradation rather than
conclusive evidence of statistical superiority.

Uniform P4 checkpoint SHA256:
`f4b66a55beca6fa5f915a6dd7b0bfe6f76d40c1a8c01f07ad3cbaaa1f4e39795`.

## Tail behavior

Uniform P4 losses relative to Greedy:

| Seed | Delta vs Greedy € | Vent t |
|---|---:|---:|
| 8100015 | +53,968.20 | 0 |
| 8100009 | +31,953.66 | 2,523.51 |
| 8100019 | +21,595.09 | 0 |

The worst economic loss is smaller than the original P4 worst loss
(`+€114,163.50`) and the original P3 worst loss (`+€250,667.04`).

Selected diagnostic seeds under uniform P4:

| Seed | Total cost € | Delta vs Greedy € | Vent t | Stored t |
|---|---:|---:|---:|---:|
| 8100002 | 1,702,854.09 | -180,798.97 | 0 | 102,524.93 |
| 8100007 | 1,666,344.09 | -34,783.47 | 406.81 | 95,360.90 |
| 8100017 | 2,025,130.79 | -229,428.42 | 3,674.90 | 111,985.36 |

Uniform P4 improves on uniform P3 in 14 of 20 seeds and lowers mean cost by
`€27,401.56`, but it is still not seed-wise monotonic. Its largest regression
relative to uniform P3 is `+€206,479.08` on seed 8100011.

## Conclusion

Using margin `0.40` consistently without a low-margin exploration bank does not
cause a large final-model degradation in this model-seed-0 experiment. P2 and P3
are slightly worse, while P4 is materially better than the original P4 and
slightly better in mean cost than the original selected P3. Protocol consistency
therefore appears to help the final aggregation round.

The experiment does not prove stability across model initializations, and
uniform margin does not eliminate non-monotonic per-seed behavior. Model seeds 1
and 2 would be required before promoting uniform P4 as a stable replacement.

## Retrospective formal-test baseline

After Iterative Q v2 had been frozen and evaluated, the model-seed-0 P3 and P4
checkpoints were run once on the same formal-test seeds 9000001–9000030. P3
achieved mean cost `€1,898,878` and 22/30 wins versus Greedy; P4 achieved
`€1,851,553` and 27/30 wins. The corresponding v2 mean was `€1,842,876`.

The paired v2-minus-P3 difference was `−€56,001`, with 95% bootstrap CI
`[−€125,019, −€619]`. The v2-minus-P4 difference was only `−€8,677`, with CI
`[−€46,595, +€30,077]`. Thus v2 is supported as better than P3, but is only
statistically tied with P4 on this test set. This retrospective comparison was
not used for model selection. See
`../iterative_q_v2_anchor_p4_ablation/FORMAL_TEST_V1_COMPARISON.md`.
