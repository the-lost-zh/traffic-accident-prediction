# Refactor Context

## Step 1: Package Skeleton And Shared Utilities

Created a new importable package under `src/traffic_accident/` while leaving legacy scripts untouched. The new package currently contains:

- `traffic_accident.utils.seed`: reproducibility helper.
- `traffic_accident.utils.device`: torch device selection.
- `traffic_accident.utils.io`: JSON, pickle, and directory helpers.
- `traffic_accident.utils.paths`: run directory layout helpers.

Planned output convention:

```text
outputs/runs/<run_id>/
├── checkpoints/
├── metrics/
├── figures/
└── artifacts/
```

## Step 2: Leakage-Safe Tabular Preprocessing

Added `traffic_accident.preprocessing.tabular` with:

- `TabularPreprocessor`: fit-on-train preprocessing for tabular accident data.
- `TabularDataBundle`: typed container for train, validation, test arrays and metadata.
- `load_tabular_data`: CSV loading, stratified train/validation/test split, train-only fit, validation/test transform, optional artifact saving.

Important behavior:

- Splits data before fitting preprocessing statistics.
- Drops columns whose training-set missing ratio exceeds the threshold.
- Keeps numeric columns and low-cardinality categorical columns.
- Uses training-set median for numeric imputation.
- Uses training-set mode for categorical imputation.
- Uses `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` so unseen categories in validation/test or production do not crash inference.
- Saves `preprocessor.pkl` and `preprocessor_metadata.json` when an artifact directory is provided.

## Validation

Commands run from project root:

```bash
python -m unittest tests.test_tabular_preprocessing
python -m compileall -q src tests
```

Both passed.

## Step 9: Non-Paired Multimodal Data And Trainer

Added:

- `traffic_accident.data.multimodal.ModalityFeatureDataset`
- `traffic_accident.data.multimodal.make_modality_loaders`
- `traffic_accident.data.multimodal.round_robin_modality_batches`
- `traffic_accident.training.multimodal.MultimodalTrainingConfig`
- `traffic_accident.training.multimodal.MultimodalTrainingResult`
- `traffic_accident.training.multimodal.UnpairedMultimodalTrainer`

Training behavior:

- Each modality owns an independent feature matrix and label vector.
- Samples do not need to be paired across modalities.
- One training step consumes one modality batch: `(modality_name, features, labels)`.
- Batches are cycled in round-robin order across available modalities.
- The model's modality-specific projector handles the modality input dimension.
- The shared Transformer backbone and shared classification head receive gradients from every modality.

Current scope:

- Inputs are already extracted feature vectors or embeddings.
- Raw image/text encoders are intentionally not added yet; they can be wrapped later before the modality projector.
- Alignment losses are optional and disabled by default, so the shared supervised baseline remains unchanged unless configured.

Validation:

```bash
python -m unittest tests.test_models tests.test_multimodal_model tests.test_multimodal_training tests.test_tabular_preprocessing tests.test_training tests.test_train_tabular_cli tests.test_evaluate_tabular_cli tests.test_inference_and_api
python -m compileall -q src api tests
```

Both passed.

Next target:

- Add a formal multimodal CLI that consumes `configs/multimodal_unpaired.yaml`.

## Notes For Next Step

The old code still works as before. The next low-risk refactor target is the training and evaluation logic inside `src/task3_classifier.py`, which should be split into:

- `training/trainer.py`
- `evaluation/metrics.py`
- `evaluation/plots.py`

The key behavior to preserve is `train_classifier(...)`, but checkpoint paths should stop being hard-coded to `models/best_model.pth`.

## Step 3: Model Package

Added `traffic_accident.models` with:

- `linear.py`: `LinearClassifier`
- `mlp.py`: `MLPClassifier`
- `transformer.py`: `TransformerClassifier`
- `ft_transformer.py`: `NumericalFeatureTokenizer`, `FTTransformerClassifier`
- `factory.py`: `create_model(...)`

The new model package is tested independently from the legacy training script. It currently mirrors the existing model behavior but removes plotting, training, path, and metric concerns from model files.

Validation added:

```bash
python -m unittest tests.test_models tests.test_tabular_preprocessing
python -m compileall -q src tests
```

Both passed.

## Step 4: Training And Metrics Modules

Added:

- `traffic_accident.evaluation.metrics.calculate_metrics`
- `traffic_accident.training.trainer.TrainingConfig`
- `traffic_accident.training.trainer.TrainingResult`
- `traffic_accident.training.trainer.SupervisedTrainer`

The new trainer owns only the supervised training loop, validation loop, prediction, checkpoint save/load, and result JSON saving. It does not create models, plot figures, run SHAP, or decide project paths. Callers pass the checkpoint path explicitly, so the old hard-coded `models/best_model.pth` behavior is avoided in new code.

Validation added:

```bash
python -m unittest tests.test_models tests.test_tabular_preprocessing tests.test_training
python -m compileall -q src tests
```

Both passed.

Next target:

- Add a prediction/evaluation loader that consumes a completed run directory.
- Keep the old `src/train.py` as a compatibility wrapper until the new CLI is verified on the real CSV.

## Step 5: New Tabular Training CLI

Added `traffic_accident.cli.train_tabular`.

The new CLI wires together:

- `load_tabular_data`
- `create_model`
- `SupervisedTrainer`
- explicit run directory layout

It writes:

```text
<output_root>/<run_name or timestamp>/
├── checkpoints/best.pt
├── metrics/final_results.json
├── artifacts/preprocessor.pkl
├── artifacts/preprocessor_metadata.json
└── config.json
```

Example command from the project root:

```bash
$env:PYTHONPATH="src"
python -m traffic_accident.cli.train_tabular --data_path data/US_Accidents_March23.csv --model_type mlp --epochs 10 --batch_size 128
```

Validation added:

```bash
python -m unittest tests.test_models tests.test_tabular_preprocessing tests.test_training tests.test_train_tabular_cli
python -m compileall -q src tests
```

Both passed. The CLI test uses a synthetic CSV and one epoch; its accuracy is not meaningful. It validates the end-to-end path and artifact creation.

## Step 6: Run-Based Evaluation CLI

Added `traffic_accident.cli.evaluate_tabular`.

The evaluation CLI loads a completed run directory instead of reconstructing model details manually. It consumes:

- `config.json`
- `artifacts/preprocessor.pkl`
- `checkpoints/best.pt`

It recreates the saved test split using the training split seed and test ratio, transforms it with the saved preprocessor, loads the model checkpoint, computes metrics, and writes:

```text
<run_dir>/metrics/evaluation.json
```

Example:

```bash
$env:PYTHONPATH="src"
python -m traffic_accident.cli.evaluate_tabular --run_dir outputs/runs/<run_id>
```

Validation added:

```bash
python -m unittest tests.test_models tests.test_tabular_preprocessing tests.test_training tests.test_train_tabular_cli tests.test_evaluate_tabular_cli
python -m compileall -q src tests
```

Both passed. This establishes the pattern that the future Flask API should also load from a run directory rather than hard-coding feature count or model type.

## Step 7: Run-Based Inference And API

Added:

- `traffic_accident.inference.tabular.TabularRunPredictor`
- `tests/test_inference_and_api.py`

Updated:

- `TabularPreprocessor.transform_features(...)` now supports feature-only inference records.
- Missing inference fields are filled with training-set preprocessing defaults.
- `api/app.py` now prefers `TRAFFIC_ACCIDENT_RUN_DIR`.

New API behavior:

- If `TRAFFIC_ACCIDENT_RUN_DIR` points to a valid run directory, Flask loads `config.json`, `artifacts/preprocessor.pkl`, and `checkpoints/best.pt`.
- If the env var is missing or invalid, Flask keeps a simple fallback predictor so the frontend can still receive a response.
- The response keeps the legacy `severity` field as a zero-based label for frontend compatibility and also returns `severity_level` as 1-4.

API startup example:

```bash
$env:PYTHONPATH="src"
$env:TRAFFIC_ACCIDENT_RUN_DIR="outputs/runs/<run_id>"
python api/app.py
```

Validation:

```bash
python -m unittest tests.test_models tests.test_tabular_preprocessing tests.test_training tests.test_train_tabular_cli tests.test_evaluate_tabular_cli tests.test_inference_and_api
python -m compileall -q src api tests
```

Both passed.

Next target:

- Add multimodal dataset/sampler abstractions for cycling unpaired modality batches.
- Add optional alignment losses such as label prototypes, supervised contrastive loss, CORAL, or MMD.

## Step 8: Non-Paired Multimodal Model Skeleton

Added `traffic_accident.models.multimodal`.

Core classes:

- `ModalityConfig`
- `ModalityProjector`
- `SharedTransformerBackbone`
- `UnifiedMultimodalTransformerClassifier`

Design:

- Each modality has its own projector from native feature dimension to shared `d_model`.
- Every modality then uses the same Transformer backbone.
- Every modality uses the same classification head.
- Forward calls use one modality at a time: `model("tabular", x)` or `model("image", x)`.
- This matches non-paired multimodal training where batches can come from separate modality datasets as long as labels share the same class space.

Current assumption:

- Image/text inputs are represented as pre-extracted feature vectors or embeddings. Raw image/text encoders can be added later behind the same projector interface.

Validation:

```bash
python -m unittest tests.test_models tests.test_multimodal_model tests.test_tabular_preprocessing tests.test_training tests.test_train_tabular_cli tests.test_evaluate_tabular_cli tests.test_inference_and_api
python -m compileall -q src api tests
```

Both passed.

## Step 10: Compatibility Wrapper, Configs, And Multimodal Alignment

Added:

- `src/train.py` is now a compatibility wrapper around `traffic_accident.cli.train_tabular`.
- `traffic_accident.config.load_config` and `flatten_cli_config` for JSON/YAML config files.
- `configs/tabular_mlp.yaml` as a standard tabular training config.
- `configs/multimodal_unpaired.yaml` as the planned unpaired multimodal training config shape.
- Legacy docstrings on `src/task1_shap_analysis.py`, `src/task2_gan.py`, and `src/task3_classifier.py`.

Updated:

- `traffic_accident.cli.train_tabular` accepts `--config`; explicit CLI flags override config values.
- `UnpairedMultimodalTrainer` supports optional label prototype alignment and supervised contrastive loss through `MultimodalTrainingConfig`.
- `UnifiedMultimodalTransformerClassifier` exposes `embedding_dim` and `num_classes` metadata for auxiliary losses.

Validation target:

```bash
python -m unittest tests.test_config_and_legacy_wrapper tests.test_multimodal_training
python -m unittest tests.test_models tests.test_multimodal_model tests.test_multimodal_training tests.test_tabular_preprocessing tests.test_training tests.test_train_tabular_cli tests.test_evaluate_tabular_cli tests.test_inference_and_api tests.test_config_and_legacy_wrapper
python -m compileall -q src api tests
```

## Step 11: Raw Text And Image Feature Entrypoints

Added:

- `traffic_accident.features.text.extract_text_features`
- `traffic_accident.features.image.extract_image_features`
- `traffic_accident.features.labels.encode_labels`
- `traffic_accident.cli.extract_features`

The CLI writes modality feature artifacts that can be consumed by the unpaired multimodal data layer:

```text
<output_dir>/
鈹溾攢鈹€ features.npy
鈹溾攢鈹€ labels.npy
鈹溾攢鈹€ metadata.json
鈹斺攢鈹€ vectorizer.pkl   # text only
```

Examples:

```bash
$env:PYTHONPATH="src"
python -m traffic_accident.cli.extract_features text --input_csv data/text.csv --text_column Description --label_column Severity --output_dir features/text
python -m traffic_accident.cli.extract_features image --input_csv data/images.csv --image_path_column image_path --label_column Severity --image_root data/images --output_dir features/image
```

Validation target:

```bash
python -m unittest tests.test_feature_extraction
python -m compileall -q src api tests
```

## Step 12: Stronger Text/Image Encoders

Added `traffic_accident.features.vision_language` for CLIP/SigLIP-style feature extraction through Hugging Face Transformers.

Rationale:

- CLIP and SigLIP produce text and image embeddings in a shared latent space.
- This matches the existing unpaired multimodal classifier better than TF-IDF text features plus RGB histogram image features.
- The lightweight encoders are still available for offline tests and quick smoke runs.

New CLI examples:

```bash
$env:PYTHONPATH="src"
python -m traffic_accident.cli.extract_features text --encoder siglip --input_csv data/text.csv --text_column Description --label_column Severity --output_dir features/text
python -m traffic_accident.cli.extract_features image --encoder siglip --input_csv data/images.csv --image_path_column image_path --label_column Severity --image_root data/images --output_dir features/image
```

Default strong encoder mapping:

```text
siglip -> google/siglip-base-patch16-224
clip   -> openai/clip-vit-large-patch14
```

Notes:

- First use downloads model weights through `transformers.from_pretrained`.
- `configs/multimodal_unpaired.yaml` now assumes 768-dimensional SigLIP text/image features.
- Use `--encoder tfidf` for text or `--encoder color` for images to keep the previous lightweight behavior.

## Step 13: FT-Transformer Tabular Projector

Updated `traffic_accident.models.multimodal` so the `tabular` modality defaults to an FT-Transformer-style projector instead of the old single-token MLP projector.

Behavior:

- `tabular` uses `TabularFTTransformerProjector` by default.
- Each tabular column is tokenized independently with a learned weight and bias.
- A small modality-local Transformer encoder models feature interactions before the shared multimodal Transformer.
- `image` and `text` still default to the dense MLP projector because SigLIP/CLIP embeddings are already single semantic vectors.
- Projector type can be overridden with `projector_type: mlp` or `projector_type: fttransformer`.

Token flow:

```text
tabular [batch, n_features]
  -> FT-style feature tokenizer [batch, n_features, d_model]
  -> tabular feature Transformer [batch, n_features, d_model]
  -> shared multimodal Transformer
  -> shared classifier
```

`configs/multimodal_unpaired.yaml` now records:

```yaml
modalities:
  tabular:
    projector_type: fttransformer
  image:
    projector_type: mlp
  text:
    projector_type: mlp
```
