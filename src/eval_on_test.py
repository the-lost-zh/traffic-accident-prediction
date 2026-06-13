import os
import numpy as np
import torch

from data_preprocessing import DataPreprocessor
from task1_shap_analysis import run_shap_analysis
from task3_classifier import create_model, get_default_config
from utils import (
    get_device,
    load_model,
    calculate_metrics,
    print_metrics,
    ensure_dir,
)


# ================== 配置区域（根据你训练时的设置修改） ==================
# 数据路径（与 train.py 默认一致）
DATA_PATH = "../data/US_Accidents_March23.csv"

# 输出与模型目录（与 train.py 默认一致）
OUTPUT_DIR = "../results"

# 模型类型：需要与训练时 --model_type 一致：linear / mlp / transformer
MODEL_TYPE = "mlp"

# 训练时保存的模型路径（train_classifier 中的默认命名）
MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_model.pth")

# 是否使用 SHAP 选择的特征：
# - 如果训练时加了 --use_selected_features 且没改 top_features/shap_model_type，
#   建议保持为 True，并保持下面两个参数一致。
# - 如果训练时没用 SHAP 选特征，把 USE_SELECTED_FEATURES 改为 False 即可。
USE_SELECTED_FEATURES = True
TOP_FEATURES = 15          # 对应训练时的 --top_features
SHAP_MODEL_TYPE = "rf"     # 对应训练时的 --shap_model_type（rf / nn）


def main():
    print("=" * 70)
    print("加载已训练模型，在测试集上评估")
    print("=" * 70)
    print(f"数据路径       : {DATA_PATH}")
    print(f"模型类型       : {MODEL_TYPE}")
    print(f"模型权重路径   : {MODEL_PATH}")
    print(f"输出目录       : {OUTPUT_DIR}")
    print(f"使用 SHAP 特征 : {USE_SELECTED_FEATURES}")
    print("=" * 70)

    device = get_device()
    ensure_dir(OUTPUT_DIR)

    # ========= 1. 预处理并划分数据（与训练时保持一致） =========
    print("\n[1/3] 数据预处理与划分 (训练/验证/测试)")
    preprocessor = DataPreprocessor(DATA_PATH)
    data = preprocessor.preprocess()

    X_test = data["X_test"]
    y_test = data["y_test"]

    # ========= 2. 根据训练时的设置选择特征（可选） =========
    if USE_SELECTED_FEATURES:
        print("\n[2/3] 运行 SHAP 分析以获取相同的特征子集")
        shap_results = run_shap_analysis(
            data,
            top_n=TOP_FEATURES,
            model_type=SHAP_MODEL_TYPE,
            output_dir=OUTPUT_DIR,
        )
        feature_indices = shap_results["feature_indices"]
        X_test = X_test[:, feature_indices]
        print(f"使用 SHAP 选择的特征数量: {len(feature_indices)}")
    else:
        print("\n[2/3] 不使用 SHAP 特征，直接使用全部特征")

    # ========= 3. 构建同结构模型，加载权重，并在测试集上评估 =========
    print("\n[3/3] 构建模型并加载已训练权重")
    config = get_default_config()
    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))

    print(f"输入维度       : {input_dim}")
    print(f"类别数         : {num_classes}")

    model = create_model(MODEL_TYPE, input_dim, num_classes, config)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型权重文件: {MODEL_PATH}\n"
            f"请先运行训练脚本，例如:\n"
            f"  cd src\n"
            f"  python train.py --model_type {MODEL_TYPE} --use_selected_features"
        )

    model = load_model(model, MODEL_PATH, device)
    model.eval()

    print("\n在测试集上进行预测与评估...")
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)

    y_pred = y_pred.cpu().numpy()

    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "测试集性能（加载已训练模型）")

    print("评估完成。")


if __name__ == "__main__":
    main()

