# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

- Add ColQwen3 and BiQwen3 support (model + processor).
- Add regression tests for `ColPaliProcessor` to validate Transformers v5 modality registration and fallback loading behavior when a processor bundle is incomplete.

### Changed

- Bump runtime compatibility to `transformers>=5.0.0,<6.0.0`, `peft>=0.18.0,<0.19.0`, and `accelerate>=1.1.0,<2.0.0`.
- Update supported Python versions to `>=3.10,<3.15` and align CI workflows to Python 3.10–3.14.
- Update all affected processor subclasses (`Qwen2/Qwen2.5/Qwen3`, `Gemma3`, `Idefics3`, `ModernVBert`, `Qwen2.5 Omni`) to explicit `__init__` modality signatures required by Transformers v5 `ProcessorMixin`.

### Fixed

- Fix ColPali/PaliGemma model loading under Transformers v5 by adapting wrapper internals to new module layout and tied-weights expectations.
- Fix ColPali processor loading for checkpoints without a complete processor bundle by explicitly falling back to `AutoImageProcessor` + `AutoTokenizer`.
- Fix ColPali collator image token id lookup to use `convert_tokens_to_ids`, compatible with Transformers v5 tokenizer backend changes.
- Fix test collection on Python 3.14 by making `tests` an explicit package (`tests/__init__.py`).
- Fix CI formatting failure by applying `ruff format` to updated ColPali processing tests.
- Fix ColQwen2 and ColQwen2.5 initialization across Transformers versions by resolving hidden size from either `config.hidden_size` or `config.text_config.hidden_size`.
- Call `post_init()` in ColIdefics3 and ColModernVBert to align model initialization with Transformers v5 expectations.
- Improve `VisualRetrieverCollator` image token id resolution by preferring processor-level `image_token_id` when available.
- Fix ColQwen2 and ColQwen2.5 LoRA checkpoint key remapping for `custom_text_proj` (`base_model.model.*` -> model keys) to avoid missing/unexpected adapter keys at load time.
- Fix ColPali LoRA adapter key remapping for `custom_text_proj` (`base_model.model.*` -> model keys) and ignore expected missing `model.lm_head.weight` during load.
- Fix ColModernVBert LoRA adapter key remapping for `custom_text_proj` (`base_model.model.*` -> model keys) to avoid missing/unexpected adapter keys at load time.
- Fix ColQwen2.5-Omni LoRA adapter key remapping for `custom_text_proj` (`base_model.model.*` -> model keys) to avoid missing/unexpected adapter keys at load time.
- Fix ColQwen3 LoRA adapter key remapping for `custom_text_proj` (`base_model.model.*` -> model keys) to avoid missing/unexpected adapter keys at load time.
- Fix ColGemma3 LoRA adapter key remapping for `custom_text_proj` (`base_model.model.*` -> model keys) to avoid missing/unexpected adapter keys at load time.
- Ensure adapter loading remains robust across Transformers v5 base-load and PEFT adapter-load code paths, preventing silent fallback to randomly initialized projection adapters in retrieval models.

### Tests

- Cover ColQwen3 processing and modeling with slow integration tests.
- Run targeted non-slow processing tests for Gemma3, Idefics3, ModernVBert, Qwen2, Qwen2.5 and Qwen3 after the Transformers v5 processor-signature migration.
- Run slow ColPali model-loading and query-forward integration tests under Transformers v5 to validate end-to-end loading behavior.
- Expand adapter checkpoint key remapping regression tests to cover ColPali, ColGemma3, ColQwen2, ColQwen2.5, ColQwen3, ColQwen2.5-Omni and ColModernVBert, including registry-backed conversion checks where needed.

## [0.3.13] - 2025-11-15

### Added

- Add ModernVBERT to the list of supported models

### Fixed

- Fix multi hard negatives training
- Fix multi dataset sampling in order to weight probability of being picked by the size of the dataset

### Changed 

- Bump transformer, torch and peft support

## [0.3.12] - 2025-07-16

### Added
- Video processing for ColQwen-Omni

### Fixed
- Fixed loading of PaliGemma and ColPali checkpoints (bug introduced in transformers 4.52)
- Fixed loading of SmolVLM (Idefics3) processors that didn't transmit image_seq_len (bug introduced in transformers 4.52)

## [0.3.11] - 2025-07-04

### Added

- Added BiIdefics3 modeling and processor.
- [Breaking] (minor) Remove support for context-augmented queries and images
- Uniform processor docstring
- Update the collator to align with the new function signatures
- Add a `process_text` method to replace the `process_query` one. We keep support of the last one for the moment, but we'll deprecate it later
- Introduce the ColPaliEngineDataset and Corpus class. This is to delegate all data loading to a standard format before training. The concept is for users to override the dataset class if needed for their specific usecases.
- Added smooth_max option to loss functions
- Added weighted in_batch terms for losses with hard negatives
- Added an option to filter out (presumably) false negatives during online training
- Added a training script in pure torch without the HF trainer
- Added a sampler to train with multiple datasets at once, with each batch coming from the same source. (experimental, might still need testing on multi-GPU)
- Adds score normalization to LI models (diving by token length) for betetr performance with CE loss
- Add experimental PLAID support

### Changed

- Stops pooling queries between GPUs and instead pools only documents, enabling training with way bigger batch sizes. We recomment training with accelerate launch now.
- Updated loss functions for better abstractions and coherence between the various loss functions. Small speedups and less memory requirements.


## [0.3.10] - 2025-04-18

### Added

- Add `LambdaTokenPooler` to allow for custom token pooling functions.
- Added training losses with negatives to InfoNCE type losses

### Changed

- Fix similarity map helpers for ColQwen2 and ColQwen2.5.
- [Breaking] (minor) Remove support for Idefics2-based models.
- Disable multithreading in `HierarchicalTokenPooler` if `num_workers` is not provided or is 1.
- [Breaking] (minor) Make `pool_factor` an argument of `pool_embeddings` instead of a `HierarchicalTokenPooler` class attribute
- Bump dependencies for transformers, torch, peft, pillow, accelerate, etc...

## [0.3.9] - 2025-04-03

### Added

- Allow user to pass custom textual context for passage inference
- Add ColQwen2.5 support and BiQwen2.5 support
- Add support for token pooling with `HierarchicalTokenPooler`.
- Allow user to specify the maximum number of image tokens in the resized images in `ColQwen2Processor` and `ColQwen2_5_Processor`.

### Changed

- Warn about evaluation being different from Vidore, and do not store results to prevent confusion.
- Remove duplicate resize code in `ColQwen2Processor` and `ColQwen2_5_Processor`.
- Simplify sequence padding for pixel values in `ColQwen2Processor` and `ColQwen2_5_Processor`.
- Remove deprecated evaluation (`CustomRetrievalEvaluator`) from trainer
- Refactor the collator classes
- Make `processor` input compulsory in `ColModelTrainingConfig`
- Make `BaseVisualRetrieverProcessor` inherit from `ProcessorMixin`
- Remove unused `tokenizer` field from `ColModelTrainingConfig`
- Bump transformers to `4.50.0` and torch to `2.6.0` to keep up with the latest versions. Note that this leads to errors on mps until transformers 4.50.4 is released.

## [0.3.8] - 2025-01-29

### Fixed

- Fix peft version in `colpali-engine[train]`
- Loosen upper bound for `accelerate`

### Tests

- Reorganize modeling tests
- Add test for ColIdefics3 (and ColSmol)

## [0.3.7] - 2025-01-28

### Changed

- Bump transformers to `4.47` to support `colSmol-256M` and `colSmol-500M`

### Fixed

- Fix checkpoints used for ColQwen2 tests

## [0.3.6] - 2025-01-10

### Added

- Add expected scores in ColPali E2E test

### Changed

- Loosen package dependencies

## [0.3.5] - 2024-12-13

### Added

- Added support for Idefics3 (and SmolVLM)

### Fixed

- Fix typing for `processor.score_multi_vector` (allow for both list and tensor inputs). This does not change how the scores are computed.
- Fix `tear_down_torch` when used on a non-MPS machine

## [0.3.4] - 2024-11-07

### Added

- General `CorpusQueryCollator` for BEIR style dataset training or hard negative training. This deprecates `HardNegCollator` but all changes to the training loop are made for a seemless update.

### Changed

- Updates BiPali config files
- Removed query augmentation tokens from BiQwen2Processor
- Modified XQwen2Processor to place `<|endoftext|>` token at the end of the document prompt (non-breaking for ColQwen but helps BiQwen).
- Removed `add_suffix` in the VisualRetrieverCollator and let the `suffix` be added in the individual processors.
- Changed the incorrect `<pad>` token to `<|endoftext|>` fo query augmentation `ColQwen2Processor`. Note that previous models were trained with `<|endoftext|>` so this is simply a non-breaking inference upgrade patch.

## [0.3.3] - 2024-10-29

### Added

- Add BiQwen2 model

### Changed

- Modified ColQwen and BiQwen to prevent the useless forward pass in the last layer of the original model (classification head)
- Bumped "breaking" dependencies on MTEB and Transformers version and made the corresponding changes in the code
- Casted Image dtype in ColPali due to breaking 4.46 transformers update
- Added a "num_image_tokens" kwarg to the `ColQwen2Processor` to allow for different image resolutions

### Fixed

- Fix wrong variable name for `ColPaliProcessor`'s prefixes

## [0.3.2] - 2024-10-17

### Added

- Restore, refactor, and improve `interpretability` module for generating similarity maps

### Changed

- Remove dummy image from `ColPaliProcessor.process_queries`

### Fixed

- Fix the `compute_hardnegs.py` script

### Tests

- Add missing `model.eval()` in tests
- Add tests for ColQwen2

## [0.3.1] - 2024-09-27

### Added

- Add module-level imports for collators
- Add sanity check in the run inference example script
- Add E2E test for ColPali
- Add Qwen2-VL support

### Changed

- Improve code clarity the run inference example script
- Subset the example dataset in the run inference example script
- Rename scorer test to `test_processing_utils`
- Greatly simplify routing logic in Trainer selection and when feeding arguments to the model forward pass (refacto)
- Removed class `ContrastiveNegativeTrainer` which is now just integrated in ContrastiveTrainer. This should not affect the user-facing API.
- Bumped transformers version to 4.45.0 to get Qwen2-VL support

### Fixed

- Import HardNegCollator at module-level if and only if datasets is available
- Remove the need for `typer` in the run inference example script
- Fix edge case when empty suffix `""` given to processor
- Fix bug in HardNegCollator since 0.3.0

## [0.3.0] - 2024-09-10

✨ This release is an exhaustive package refacto, making ColPali more modular and easier to use.

🚨 It is **NOT** backward-compatible with previous versions.

### Added

- Restructure the `utils` module
- Restructure the model training code
- Add custom `Processor` classes to easily process images and/or queries
- Enable module-level imports
- Add scoring to processor
- Add `CustomRetrievalEvaluator`
- Add missing typing
- Add tests for model, processor, scorer, and collator
- Lint `Changelog`
- Add missing docstrings
- Add "Ruff" and "Test" CI pipelines

### Changed

- Restructure all modules to closely follow the [`transformers`](https://github.com/huggingface/transformers) architecture
- Hugely simplify the collator implementation to make it model-agnostic
- `ColPaliProcessor`'s `process_queries` doesn't need a mock image input anymore
- Clean `pyproject.toml`
- Loosen the required dependencies
- Replace `black` with the `ruff` linter

### Removed

- Remove `interpretability` and `eval_manager` modules
- Remove unused utils
- Remove `TextRetrieverCollator`
- Remove `HardNegDocmatixCollator`

### Fixed

- Fix wrong PIL import
- Fix dependency issues

## [0.2.2] - 2024-09-06

### Fixed

- Remove forced "cuda" usage in Retrieval Evaluator

## [0.2.1] - 2024-09-02

Patch query preprocessing helper function disalignement with training scheme.

### Fixed

- Add 10 extra pad token by default to the query to act as reasoning buffers. This was added in the collator but not the external helper function for inference purposes.

## [0.2.0] - 2024-08-29

Large refactoring to adress several issues and add features. This release is not backward compatible with previous versions.
The models trained under this version will exhibit degraded performance if used with the previous version of the code and vice versa.

[Branch](https://github.com/illuin-tech/colpali/pull/23)

### Added

- Added multiple training options for training with hard negatives. This leads to better model performance !
- Added options for restarting training from a checkpoint.

### Changed

- Optionally load ColPali models from pre-initialized backbones of the same shape to remove any stochastic initialization when loading adapters. This fixes [11](https://github.com/illuin-tech/colpali/issues/11) and [17](https://github.com/illuin-tech/colpali/issues/17).

### Fixed

- Set padding side to right in the tokenizer to fix misalignement issue between different query lengths in the same batch. Fixes [12](https://github.com/illuin-tech/colpali/issues/12)
- Add 10 extra pad token by default to the query to act as reasoning buffers. This enables the above fix to be made without degrading performance and cleans up the old technique of using `<unused>` tokens.

## [0.1.1] - 2024-08-28
  
Minor patch release to fix packaging issues.

### Fixed

- [Branch](https://github.com/illuin-tech/colpali/commit/bd55e88c7af7069dde943f00665181fb94631cdd)
  Fix .gitignore to include all necessary files in the package.

## [0.1.0] - 2024-08-28

Initial code release corresponding to the paper.
