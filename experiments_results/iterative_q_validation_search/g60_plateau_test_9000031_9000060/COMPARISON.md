# G60-P4 vs Recursive Plateau-20/50 P4 vs Uniform-Margin40 P4

Paired evaluation on seeds `9000031–9000060` (30 episodes). Lower total cost is better. All models use the same Greedy baseline.

## Model-level results

| Model | Mean total cost | Saving vs Greedy | W/T/L vs Greedy | Vented (t) | Stored (t) | Overrides |
|---|---:|---:|---:|---:|---:|---:|
| G60-P4 | €1,870,036.72 | 17.043% | 27/0/3 | 864.68 | 110,449.80 | 11.43 |
| Recursive Plateau-20/50 P4 | €1,839,562.99 | 18.394% | 28/0/2 | 761.60 | 110,458.25 | 10.53 |
| Uniform-Margin40 P4 | €1,874,063.75 | 16.864% | 26/0/4 | 1,061.32 | 110,995.93 | 11.20 |

Shared mean Greedy cost: €2,254,211.56.

## Paired model comparisons

Cost difference is `left − right`; a negative value favors the left model.

| Left model | Right model | Mean difference | Bootstrap 95% CI | Left better/tie/worse |
|---|---|---:|---:|---:|
| G60-P4 | Uniform-Margin40 P4 | −€4,027 | [−€54,161, +€45,261] | 15/0/15 |
| Recursive Plateau-20/50 P4 | Uniform-Margin40 P4 | −€34,501 | [−€101,366, +€36,347] | 17/0/13 |
| Recursive Plateau-20/50 P4 | G60-P4 | −€30,474 | [−€78,234, +€18,719] | 19/0/11 |

## Regressions versus Greedy

- G60-P4: `9000044` (€122,881.44), `9000047` (€6,645.15), `9000038` (€4,929.85).
- Recursive Plateau-20/50 P4: `9000044` (€52,722.78), `9000047` (€9,551.13).
- Uniform-Margin40 P4: `9000041` (€149,798.85), `9000032` (€86,187.05), `9000044` (€48,986.52), `9000046` (€3,801.70).

## Largest paired regressions of the left model

- G60-P4 vs Uniform-Margin40 P4: `9000036` (€356,276.60), `9000054` (€231,144.56), `9000057` (€165,921.39), `9000035` (€129,938.82), `9000056` (€100,048.74).
- Recursive Plateau-20/50 P4 vs Uniform-Margin40 P4: `9000054` (€591,794.31), `9000052` (€340,146.54), `9000057` (€82,708.27), `9000050` (€46,955.81), `9000056` (€46,495.68).
- Recursive Plateau-20/50 P4 vs G60-P4: `9000054` (€360,649.75), `9000052` (€310,732.41), `9000042` (€50,953.42), `9000050` (€35,331.89), `9000049` (€18,681.30).

## Reproducibility

- Bootstrap: 200,000 paired resamples, RNG seed `20260729`.
- SLURM jobs: environment check `34189`, G60 evaluation `34190`, Plateau evaluation `34191`; all completed with exit code `0:0`.
- All three stderr logs were empty.
- Per-seed paired values: `paired_seed_comparison.csv`.
- G60-P4 checkpoint SHA-256: `e529eb06038a4842f58eb97a912ad0f72de9000f03d39d9bc8839e8690febedc`.
- Recursive Plateau-20/50 P4 checkpoint SHA-256: `8439aab87231419e3d57a67d59cd862dd55e0541044c989e6c45c34268138160`.
