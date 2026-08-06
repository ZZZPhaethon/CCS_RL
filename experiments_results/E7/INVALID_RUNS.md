# Excluded E7 Direct-global runs

- `formal_run02_direct_global` (SLURM 35636): completed with a different
  checkpoint copy from the original receding-cyclic run.
- `formal_run03_direct_global` (SLURM 35696): stopped after the first shard
  audit showed that the attempted environment-variable override had not
  changed the checkpoint path.
- `formal_run04_direct_global` (SLURM 35728): the checkpoint hash was correct,
  but the preflight shard exposed an older remote evaluation module that did
  not apply the checkpoint's excluded state feature and produced 94 instead
  of 93 state dimensions. No formal array was launched.

Neither run is included in the E7 summary. The replacement submission uses a
dedicated script that verifies the expected SHA-256 checkpoint hash inside
every array task before evaluation.
