#!/usr/bin/env bash

# 一键训练并在测试集上评估（Linux / WSL 使用）
# 用法：
#   chmod +x train.sh
#   ./train.sh
# 如需给 train.py 追加自定义参数，例如 --epochs 50：
#   ./train.sh --epochs 50

set -e

# 定位到项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"

# 1. 安装依赖（如果已安装会自动跳过已满足的包）
if [ -f "requirements.txt" ]; then
  echo "安装依赖: pip install -r requirements.txt"
  pip install -r requirements.txt
else
  echo "未找到 requirements.txt，跳过依赖安装"
fi

# 2. 进入 src 目录并训练模型
cd "$PROJECT_ROOT/src"
echo "进入目录: $(pwd)"

echo "开始训练模型 (mlp, 使用 SHAP 选择特征)..."
python train.py --model_type mlp --use_selected_features "$@"

# 3. 训练完成后，加载已训练模型，在测试集上评估
echo "训练完成，开始在测试集上评估..."
python eval_on_test.py

echo "全部流程完成。"

