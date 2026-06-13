# 交通事故损伤等级预测系统

基于PyTorch和SHAP的交通事故损伤等级预测系统，实现关键参数挖掘和智能预测。

## 项目结构

```
traffic-accident/
├── data/                           # 数据目录
│   └── US_Accidents_March23.csv    # 交通事故数据集
├── src/                            # 源代码目录
│   ├── data_preprocessing.py       # 数据预处理模块
│   ├── task1_shap_analysis.py      # 任务1：SHAP特征分析
│   ├── task2_gan.py                # 任务2：GAN模型
│   ├── task3_classifier.py         # 任务3：分类器训练
│   ├── train.py                    # 主训练脚本
│   └── utils.py                    # 工具函数
├── frontend/                       # 前端页面
│   ├── index.html                  # 主页面
│   ├── css/                        # CSS样式
│   └── js/                         # JavaScript逻辑
├── api/                            # 后端API
│   └── app.py                      # Flask应用
├── models/                         # 模型保存目录
├── notebook/                       # Jupyter Notebook
│   └── traffic_accident_analysis.ipynb
├── results/                        # 结果输出目录
├── requirements.txt                # 依赖包
└── README.md                       # 项目说明
```

## 功能特性

### 任务1：关键参数定位挖掘
- 使用SHAP值分析识别影响损伤等级的关键参数
- 支持随机森林和神经网络作为基础模型
- 可视化特征重要性

### 任务2：场景生成（GAN）
- 基于生成对抗网络生成交通事故场景数据
- 支持自定义潜在空间维度
- 实现生成器和判别器的训练

### 任务3：损伤等级预测
- 支持多种分类器：线性分类器、MLP、Transformer
- GPU加速训练
- 自动早停和学习率调整
- 完整的性能评估和可视化
- tqdm进度条实时显示训练过程
- Transformer参数兼容性检查

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 方式1：使用命令行训练

```bash
cd src
python train.py --help
```

#### 基本用法

```bash
# 使用默认参数训练线性分类器
python train.py

# 训练MLP模型
python train.py --model_type mlp
#or this code
python src/train.py --data_path "data/US_Accidents_March23.csv" --model_type mlp --batch_size 128 --epochs 10 --train_gan
# 训练Transformer模型
python train.py --model_type transformer --epochs 50
python src/train.py --data_path "data/US_Accidents_March23.csv" --model_type transformer --batch_size 128 --epochs 10 --train_gan
#默认gan是不训练的，如果不写就不会训练
# 使用SHAP选择的特征训练
python train.py --use_selected_features --top_features 15

# 自定义训练参数
python train.py --model_type mlp --epochs 100 --batch_size 256 --learning_rate 0.0005
```

#### 参数说明

- `--data_path`: 数据文件路径（默认：../data/US_Accidents_March23.csv）
- `--output_dir`: 输出目录（默认：../results）
- `--skip_shap`: 跳过SHAP分析，直接使用全部特征
- `--top_features`: SHAP选择的Top特征数量（默认：15）
- `--shap_model_type`: SHAP分析使用的模型类型（rf/nn，默认：rf）
- `--model_type`: 分类器模型类型（linear/mlp/transformer，默认：linear）
- `--epochs`: 训练轮数（默认：100）
- `--batch_size`: 批次大小（默认：512）
- `--learning_rate`: 学习率（默认：0.001）
- `--dropout`: Dropout率（默认：0.3）
- `--use_selected_features`: 使用SHAP选择的特征进行训练
- `--seed`: 随机种子（默认：42）
- `--train_gan`: 训练GAN模型
- `--gan_epochs`: GAN训练轮数（默认：100）
- `--gan_batch_size`: GAN批次大小（默认：128）
- `--gan_latent_dim`: GAN潜在空间维度（默认：100）
- `--gan_learning_rate`: GAN学习率（默认：0.0002）

### 方式2：使用Jupyter Notebook

```bash
cd notebook
jupyter notebook traffic_accident_analysis.ipynb
```

在Notebook中可以逐步执行每个模块，查看中间结果和可视化图表。

## 模型架构

### 1. 线性分类器
最简单的模型，仅包含一个线性层，适合作为基线模型。

### 2. MLP分类器
多层感知机，包含：
- 输入层
- 2个隐藏层（默认：128, 64）
- ReLU激活函数
- Dropout正则化
- 输出层，输出维度为类别数

### 3. Transformer分类器
基于Transformer的模型，包含：
- 输入投影层
- Transformer编码器（默认：1层，4个注意力头）
- 分类头，输出维度为类别数
- 参数兼容性检查（确保d_model能被nhead整除）

## 训练配置

所有模型的超参数都集中在配置字典中：

```python
config = {
    'epochs': 100,              # 训练轮数
    'batch_size': 512,          # 批次大小
    'learning_rate': 0.001,     # 学习率
    'dropout': 0.3,             # Dropout率
    'early_stopping_patience': 15,  # 早停耐心值
    'hidden_dims': [128, 64],   # MLP隐藏层维度
    'd_model': 64,              # Transformer模型维度
    'nhead': 4,                 # 注意力头数
    'num_layers': 1,            # Transformer层数
    'dim_feedforward': 128      # 前馈网络维度
}
```

## 输出结果

训练完成后，results目录会包含：

1. **模型文件**
   - `{model_type}_model.pth`: 训练好的模型权重
  - `{model_type}_training_results.json`: 训练结果
2. **可视化图表**
   - `feature_importance.png`: 特征重要性柱状图
   - `shap_summary.png`: SHAP摘要图
   - `{model_type}_confusion_matrix.png`: 混淆矩阵
   - `{model_type}_training_history.png`: 训练历史曲线
   - `model_comparison.png`: 模型性能对比图

3. **数据文件**
   - `shap_results.pkl`: SHAP分析结果
   - `feature_importance.csv`: 特征重要性CSV
   - `final_results.json`: 最终训练结果

## 性能指标

系统会计算以下性能指标：

- Accuracy（准确率）
- Precision（精确率，macro和weighted）
- Recall（召回率，macro和weighted）
- F1-score（macro和weighted）
- AUC（ROC曲线下的面积）
- Confusion Matrix（混淆矩阵）

## 技术栈

- **深度学习框架**: PyTorch
- **特征重要性**: SHAP
- **数据处理**: pandas, numpy
- **机器学习**: scikit-learn
- **可视化**: matplotlib, seaborn

## 注意事项

1. 确保数据文件位于`data/`目录下
2. 如果有GPU，系统会自动使用GPU加速
3. 首次运行SHAP分析可能需要较长时间
4. Transformer模型训练时间较长，建议先从线性分类器开始

## 预期结果

- SHAP分析：识别10-15个关键特征
- 模型准确率：≥85%（目标）
- 完整的训练和测试报告

## 扩展功能

1. 实现任务2：场景生成（GAN/扩散模型）  √
2. 添加更多模型变体                    ？
3. 实现模型集成                      ？
4. 添加Web界面进行预测               √

## 前端页面使用

### 启动后端API服务
```bash
python api/app.py
```

### 打开前端页面
- 直接打开 `frontend/index.html` 文件
- 或使用本地服务器托管前端文件

### 使用方法
1. 填写事故参数
2. 点击"预测损伤等级"按钮
3. 查看预测结果
