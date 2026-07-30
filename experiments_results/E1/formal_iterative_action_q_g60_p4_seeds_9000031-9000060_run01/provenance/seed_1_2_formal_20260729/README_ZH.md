# Model seed 1/2 正式评估 provenance

- `34433`：G60-P4 model seed 1，`COMPLETED 0:0`，耗时 00:02:22。
- `34434`：G60-P4 model seed 2，`COMPLETED 0:0`，耗时 00:02:22。
- 两个作业均评估 `9000031–9000060`，使用相同冻结 gate：
  required heads 4、margin 0.40、maximum overrides 12。
- 评估导出增加 episode 内的 fuel、conditioning、reconditioning、
  loading、unloading、vent penalty 和 storage-shortfall penalty。
- Terminal cleanup 按当前比较规范以 operating-cost 总额单独记录，不拆分为各成本分项。
