# Fresh two-output LOPO recipes

`train_models_lopo.sh` reads these five JSON files independently of `models/`
and `finished_models/`. Both output directories can be absent or empty when
training starts. Keep these recipes and the raw annotated recordings.

The recipes preserve the effective architecture and preprocessing settings of
the original finished models. They contain no checkpoints, fitted quality
statistics, selected thresholds, or evaluation results. The maximum epoch
budget is 35; it is not the old model's selected epoch.

These files supply parameter values to the current training implementation.
Training uses preprocessing contract version 4 and creates a new LSTM with
`current` and `previous` output heads. Nested LOPO selects the epoch and primary
threshold, then calibrates recovery with the selected primary threshold fixed.
The wrapper forces joint two-output training and regenerates preprocessing;
old weights, cached features, and historical version or policy metadata are
not inherited from the original models.

The wrapper overrides LR with its shared `--lr 0.001` argument and uses base
seed 42 by default. Edit the LR in `build_training_command` to adjust it for
all jobs; use `--seed` and `--num-epochs` for those overrides. Use
`--recipe-dir DIR` to load an alternative set of recipes.

New training outputs receive the current preprocessing contract version
automatically. Final models are published under
`finished_models/MODEL/recovery/` with `_recovery` model names.
