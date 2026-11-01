# Decisions log

## Environment and versions

## Datasets

## Models

- **Prediction output type.** Calling `KumoTabular` returns `sdm.TableTensor`. The API reference
  ([ICLModel.forward](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.ICLModel.html))
  documents it as a `torch.Tensor` subclass. For classification, the default recipe ends in
  `AverageEstimators()` then `Softmax()`. `TableTensor` documents no attribute or method for class probabilities, and
  `.numpy()` raises "not supported for tensor subclasses". The notebook therefore converts with
  `probs.as_subclass(torch.Tensor).detach().float().cpu().numpy()`. It prints `type(probs)` on the first
  run to confirm this; if that differs from `sdm.TableTensor`, update this entry.

## Evaluation

## Open questions
