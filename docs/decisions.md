# Decisions log

## Environment and versions

- **Local CPU debug mode.** `DEBUG_CPU` (default `False`) lets the notebook run on a Mac without an NVIDIA GPU:
  CPU, float32 (autocast off), 1,000 context rows, a stratified 200-row subset of the query set,
  `num_estimators=1`, Small only, no GPU memory measurement, output to `results/raw/smoke_test_debug.csv`.
  With the flag `False`, the GPU path is unchanged apart from the `check()` fix under Models.
  Local environment: `.venv/` (Python 3.12.15, PyTorch 2.15.0, structured-data-models 0.2.1, CPU).
  Debug run result: output (200, 7), rows sum to 1, accuracy 0.760 vs majority baseline 0.485.

## Datasets

## Models

- **Prediction output type (corrected 2026-11-01).** Calling `KumoTabular` returns
  `sdm.tensor.table.TableTensor`, confirmed by `type(probs)` in a local CPU run with structured-data-models 0.2.1.
  It is a wrapper subclass that sends every op through `__torch_dispatch__` and raises `NotImplementedError` for
  any op it doesn't handle (`sdm/tensor/table.py:1099-1111`). The earlier `.as_subclass(torch.Tensor).detach()`
  therefore failed. The library's own source shows the intended route:
  - `Softmax` writes the probabilities into the numerical block via `replace_blocks(numerical=...)`
    (`sdm/processing/output/softmax.py:36-40`).
  - The `.numerical` property returns that block as a plain `Tensor` (`sdm/tensor/table.py:866-869`).
  - The library's own `to_class_indices` reads predictions the same way, `pred.columns[Stype.numerical]` and
    `pred.numerical`, and documents that "column names define the class order"
    (`sdm/evaluation/classification.py:21-22, 80-81`).

  `check()` now uses `probs.numerical` for the values and the block's column names for the class labels,
  replacing the earlier undocumented assumption of sorted class order. It also asserts the output shape is
  query rows by 7.
- **Model revision (fixed 2026-11-01).** The library pins the weights to tag `v1.0.1` of `nvidia/Kumo-Tabular`
  (`sdm/models/kumo/tabular/model.py:158-162`). It does not download the Hub's latest commit. The notebook
  originally recorded the latest commit (`c5893d4…`) as `hf_revision`, which differed from the weights actually
  loaded (`3c3e10b…`). Fixed: the revision cell now records the commit the pinned tag points to,
  `HfApi().model_info(HF_REPO, revision="v1.0.1").sha`, and stores the tag as `HF_TAG`. Locally this resolves to
  `3c3e10bbdb590ace29e7026847f92db3c603096d`, matching the cache. `hf_cached_revisions` stays as a cross-check.
  If a library upgrade changes the pinned tag, update `HF_TAG`.

## Evaluation

## Open questions
