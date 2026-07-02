# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Traffic accident severity prediction system using multi-modal representation learning. The system classifies accident severity into 4 levels using tabular data (US_Accidents_March23.csv), with support for text and image modalities via CLIP/SigLIP encoders. Built with PyTorch, SHAP explainability, and a Flask + vanilla HTML/JS deployment.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install xgboost imbalanced-learn  # optional: baselines + SMOTE

# Train a model (cd src first — all paths in train.py are relative to src/)
cd src

# Recommended: FT-Transformer with Focal Loss (default for imbalance)
python train.py --model_type fttransformer --epochs 50

# With SMOTE oversampling
python train.py --model_type fttransformer --use_smote

# Without Focal Loss
python train.py --model_type fttransformer --no_focal_loss

# Other models
python train.py --model_type mlp
python train.py --model_type transformer --batch_size 128

# Train FT-Transformer (dedicated script with extra visualization)
python train_fttransformer.py --model_type fttransformer --epochs 50

# Evaluate (includes Majority + XGBoost baselines + per-class metrics)
python eval_on_test.py

# Multimodal training (extract features first, then train)
python multimodal/extract_features.py text --input_csv ... --output_dir ...
python train_multimodal.py --config ../configs/multimodal_unpaired.yaml

# Start the Flask API server (port 8888)
python api/app.py
# Multimodal API
MULTIMODAL_RUN_DIR=../outputs/multimodal_runs/multimodal_unpaired python api/app.py

# Run SHAP analysis standalone
python task1_shap_analysis.py

# Run tests (from project root)
python -m pytest tests/ -v

# Launch Jupyter notebook
cd notebook && jupyter notebook traffic_accident_analysis.ipynb
```

## Architecture

### Data Flow
```
CSV → DataPreprocessor (missing values → type detection → encoding → scaling → split)
     → SHAPAnalyzer (RF or SimpleNN → PermutationExplainer → top-K feature selection)
     → train_classifier (create_model → ModelTrainer → evaluate on test set)
     → PredictiveAgent (loads model + preprocessor + explainer for inference)
```

### Key Classes in `src/`

- **`DataPreprocessor`** (`data_preprocessing.py`): Full pipeline — load CSV, handle missing values (drop cols >50% missing, median/mode imputation), identify numeric vs categorical features by uniqueness ratio, LabelEncode categoricals, StandardScale numerics, stratified train/val/test split. Supports `save()`/`load()` via joblib for inference reuse.

- **`ModelTrainer`** (`task3_classifier.py`): Generic trainer with AdamW optimizer, ReduceLROnPlateau scheduler, early stopping, gradient clipping, NaN/Inf guards, class weights for imbalance, warmup epochs, and label smoothing. Call `train()` with numpy arrays — it wraps them in DataLoaders internally.

- **Model architectures** (all in `task3_classifier.py`):
  - `LinearClassifier` — baseline
  - `MLPClassifier` — configurable hidden_dims + dropout
  - `TransformerClassifier` — treats input as a single sequence token (legacy approach)
  - `FTTransformerClassifier` — **best performer**: treats each tabular feature as a token with a learnable CLS token (like ViT for tables). Uses `NumericalFeatureTokenizer` (vectorized per-feature weights) and positional encoding. Has `get_feature_importance()` via CLS-token attention.

- **`PredictiveAgent`** (`agent.py`): Orchestrates inference. Loads preprocessor + model + SHAP PermutationExplainer. `predict_with_explanation()` returns severity class, probabilities, and per-feature SHAP contributions. The API calls this; falls back to a hardcoded simulation if the agent isn't loaded.

- **`SHAPAnalyzer`** (`task1_shap_analysis.py`): Trains a base model (RF or SimpleNN), computes SHAP values via `PermutationExplainer`, selects top-N features, plots summary.

- **GAN** (`task2_gan.py`): Standard GAN with `Generator` + `Discriminator` + `GANTrainer` for generating synthetic accident scenarios.

### Entry Points

- `src/train.py` — main CLI (preprocess → train classifier → optional GAN → save agent). Default: FT-Transformer + Focal Loss
- `src/train_fttransformer.py` — dedicated FT-Transformer training with attention visualization
- `src/train_multimodal.py` — unpaired multimodal training (tabular + text + image)
- `src/eval_on_test.py` — evaluates saved model against Majority + XGBoost baselines, prints per-class metrics
- `api/app.py` — Flask server on port 8888, serves `/api/predict` (POST), `/api/predict/multimodal` (POST), `/api/health` (GET)
- `frontend/index.html` — tabular prediction + multimodal prediction (mode tabs)

### Multimodal Package (`src/multimodal/`)

- `model.py` — `UnifiedMultimodalTransformerClassifier` with per-modality projectors + shared backbone
- `trainer.py` — `UnpairedMultimodalTrainer` using round-robin modality batch cycling
- `data.py` — `ModalityFeatureDataset`, `ModalityBatch`, `make_modality_loaders`
- `vision_language.py` — CLIP/SigLIP text and image encoders
- `text_features.py`, `image_features.py` — TF-IDF, color histogram feature extraction
- `config_loader.py` — YAML/JSON config parsing

### Config System

`configs/multimodal_unpaired.yaml` defines the unpaired multi-modal fusion setup. For tabular training, use CLI args or `get_default_config()` in `task3_classifier.py`.

### Important Conventions

- **Working directory matters**: `train.py` and related scripts use paths relative to `src/` (e.g., `../data/...`, `../results`). Always `cd src` before running training scripts.
- **Severity classes**: 1-4 in the raw data, mapped to 0-3 internally by subtracting 1. The `Severity` column in the CSV uses 1-indexed values.
- **Class imbalance**: Severity 2 dominates (~79.7%), so accuracy alone is misleading. Always check macro-F1 and per-class metrics. Class weights are computed automatically in `train_classifier()`.
- **Feature selection**: SHAP analysis selects top-K features. Use `--use_selected_features` during training to train on the subset. The `feature_indices` are saved in `final_results.json`.
- **Model naming**: Saved as `{model_type}_model.pth` in the output directory. The best model during training is saved as `models/best_model.pth` (hardcoded path).
- **FT-Transformer specifics**: `d_model` must be divisible by `nhead`. The code auto-adjusts `nhead` if incompatible. FT-Transformer also enables warmup (5 epochs) and label smoothing (0.1) by default when `model_type='fttransformer'`.
