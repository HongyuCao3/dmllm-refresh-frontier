# Data Rules (applies to src/data/**)

## Status
- `src/data/` is not in the tree: the root `.gitignore` pattern `data/` also matches `src/data/`,
  so it was never committed. `configs/dataset/wine.yaml` and `tests/test_eval_mode.py` still
  import `src.data.wine_datamodule.WineDataModule`. These rules govern that layer when it is added.

## DataModule contract
- Must implement `setup(stage)` creating `train_ds`, `val_ds`, `test_ds`.
- DataLoaders returned by `train_dataloader / val_dataloader / test_dataloader`.
- Batch shape must be compatible with `model.forward` input.
- Never preprocess inline in model/lit_module; raw→tensor conversion belongs to the data layer.

## Constraints
- No dataset mixing / multi-datamodule composition.
