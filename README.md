# 交通事故损伤等级预测系统

基于多模态表示学习的交通事故损伤等级预测系统，支持表格、文本、图像三种模态，具备 SHAP 可解释性和完整部署流程。

## 项目结构

```
traffic-accident/
├── data/                              # 数据目录
│   └── US_Accidents_March23.csv       # 交通事故数据集
├── src/                               # 源代码
│   ├── data_preprocessing.py          # 数据预处理 (旧版)
│   ├── task1_shap_analysis.py         # SHAP 特征分析
│   ├── task2_gan.py                   # GAN 模型
│   ├── task3_classifier.py            # 分类器 + ModelTrainer (含 Focal Loss)
│   ├── train.py                       # 主训练脚本 (CLI)
│   ├── train_fttransformer.py         # FT-Transformer 专用训练
│   ├── train_multimodal.py            # 多模态统一训练 ⭐
│   ├── eval_on_test.py                # 评估脚本 (含 XGBoost + Majority 基线)
│   ├── agent.py                       # 表格推理智能体
│   ├── multimodal_agent.py            # 多模态推理智能体 ⭐
│   ├── utils.py                       # 工具函数 (含基线评估/SMOTE)
│   └── traffic_accident/              # 新版规范化包结构
│       ├── models/                    #   FT-Transformer, Multimodal
│       ├── training/                  #   SupervisedTrainer, UnpairedMultimodalTrainer
│       ├── features/                  #   文本/图像/表格特征提取
│       ├── preprocessing/             #   TabularPreprocessor
│       └── cli/                       #   特征提取CLI, 训练CLI
├── api/                               # Flask API
│   └── app.py                         # /api/predict + /api/predict/multimodal
├── frontend/                          # Web 前端
│   └── index.html                     # 表格预测 + 多模态预测 (模式切换)
├── configs/                           # YAML 配置文件
│   ├── tabular_mlp.yaml
│   └── multimodal_unpaired.yaml
├── models/                            # 模型保存目录
├── results/                           # 结果输出
└── tests/                             # 测试
```

## 安装

```bash
pip install -r requirements.txt

# 额外依赖 (可选)
pip install xgboost imbalanced-learn  # XGBoost 基线 + SMOTE 过采样
```

## 快速开始

### 训练

```bash
cd src

# 推荐: FT-Transformer (默认启用 Focal Loss 处理不平衡)
python train.py --model_type fttransformer --epochs 50

# 使用 SMOTE 过采样
python train.py --model_type fttransformer --use_smote

# 显式禁用 Focal Loss
python train.py --model_type fttransformer --no_focal_loss

# 自定义 Focal Loss γ 值
python train.py --model_type fttransformer --focal_loss --focal_gamma 1.5

# SHAP 预筛选特征 (不推荐 — 会丢失高阶交互信息)
python train.py --model_type fttransformer --use_selected_features

# GAN 场景生成 (不推荐 — SMOTE/Focal Loss 性价比更高)
python train.py --model_type fttransformer --train_gan
```

### 评估

```bash
# 在测试集上评估 (含 Majority 基线 + XGBoost 基线 + per-class metrics)
cd src
python eval_on_test.py

# 注意: 编辑脚本中的 MODEL_TYPE 和 MODEL_PATH 常量
```

### 多模态训练

```bash
# 1. 提取特征
cd src
python traffic_accident/cli/extract_features.py tabular \
    --input_csv ../data/US_Accidents_March23.csv --output_dir ../features/tabular

python traffic_accident/cli/extract_features.py text \
    --input_csv ../data/captions.csv --text_column "Description" \
    --label_column Severity --output_dir ../features/text

python traffic_accident/cli/extract_features.py image \
    --input_csv ../data/images.csv --image_path_column "path" \
    --label_column Severity --output_dir ../features/image

# 2. 训练
python train_multimodal.py --config ../configs/multimodal_unpaired.yaml
```

### API 部署

```bash
# 表格预测
python api/app.py

# 多模态预测
MULTIMODAL_RUN_DIR=../outputs/multimodal_runs/multimodal_unpaired python api/app.py

# 然后打开 frontend/index.html
```

### Jupyter Notebook

```bash
cd notebook && jupyter notebook traffic_accident_analysis.ipynb
```

## 模型架构

| 模型 | 适用场景 | 不平衡处理 |
|------|----------|-----------|
| **LinearClassifier** | 最简基线 | Class weights |
| **MLPClassifier** | 非线性基线 | Class weights |
| **TransformerClassifier** | 旧版 Transformer 基线 | Class weights |
| **FT-TransformerClassifier** ⭐ | 表格特征交互建模 (推荐) | Focal Loss + Class weights + SMOTE |
| **XGBoost** (基线) | 树模型 Baseline (eval 中) | scale_pos_weight |
| **UnifiedMultimodalTransformer** | 多模态融合 | Focal Loss + Prototype Alignment |

### Focal Loss

FT-Transformer 默认启用 Focal Loss (γ=2.0)，专为 Severity 2 占 79.7% 的极端不平衡设计：

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

γ=2 时，正确分类的多数类样本 loss 大幅降低 (~0.01x)，模型更关注难以分类的少数类样本。

### SHAP 使用建议

- **✅ 推荐**: 训练后对 FT-Transformer 做 SHAP 解释 (运行 `task1_shap_analysis.py`)
- **❌ 不推荐**: 训练前用 RF 的 SHAP 筛选特征 (会丢失 FT-Transformer 能捕捉的高阶交互)
- 如需特征筛选，建议对比"全特征 vs SHAP 筛选"的性能差异再决定

## 关键参数

```
--model_type     分类器 (linear/mlp/transformer/fttransformer, 默认 fttransformer)
--epochs         训练轮数 (默认 100)
--batch_size     批次大小 (默认 512)
--learning_rate  学习率 (默认 0.001)
--dropout        Dropout (默认 0.3)
--focal_loss     启用 Focal Loss (FT-Transformer 默认启用)
--no_focal_loss  禁用 Focal Loss
--focal_gamma    Focal Loss γ (默认 2.0)
--use_smote      启用 SMOTE 过采样
--skip_shap      跳过 SHAP (推荐 — SHAP 应作为训练后解释)
--use_selected_features  用 SHAP 预筛选特征 (不推荐)
```

## 评估指标

### 核心指标 (不平衡数据)

| 指标 | 说明 |
|------|------|
| **Macro-F1** | 每个类等权平均，反映少数类召回能力 (最重要) |
| Per-class Recall | 每个 Severity 级别的召回率 |
| Weighted-F1 | 按样本量加权的 F1 |

### 基线对比

`eval_on_test.py` 自动输出两个基线：

1. **Majority Baseline**: 始终预测 Severity 2 (多数类)。Accuracy ≈ 79.7%, Macro-F1 ≈ 0.11
2. **XGBoost Baseline**: 200 棵树，max_depth=8。在不平衡表格数据上通常接近或超过简单 MLP

⚠ 若模型的 Macro-F1 未超过基线，说明深度学习模型在此数据集上无增量价值。

## 输出结果

训练后 `results/` 包含:

- `{model_type}_model.pth`: 模型权重
- `final_results.json`: 完整训练结果 (含多数类基线对比)
- `{model_type}_confusion_matrix.png`: 混淆矩阵
- `{model_type}_training_history.png`: 训练曲线
- `feature_importance.png` / `shap_summary.png`: SHAP 解释图

## 数据集说明

- **来源**: US_Accidents_March23.csv (美国真实交通事故)
- **类别**: Severity 1-4 (映射为 0-3)
- **不平衡**: Severity 2 占 ~79.7%，Severity 4 极少
- **特征**: 60+ 维 (数值 + 编码后分类变量)，含经纬度、天气、能见度、路口等

## 注意事项

1. 所有训练脚本需从 `src/` 目录运行 (路径相对)
2. **Accuracy 在此数据上无参考价值** — 始终关注 Macro-F1 和 per-class Recall
3. SHAP 首次运算耗时较长，建议先在子集上测试
4. 多模态需要先运行特征提取 (CLIP/SigLIP 需要 transformers 库)
5. 模型未加载时 API 返回 503 (不会返回假数据)

## 技术栈

PyTorch · SHAP · scikit-learn · XGBoost · Flask · CLIP/SigLIP · FT-Transformer · Focal Loss · SMOTE
