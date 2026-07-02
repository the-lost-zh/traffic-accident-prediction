import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data_preprocessing import DataPreprocessor
from task1_shap_analysis import run_shap_analysis
from task3_classifier import create_model, get_default_config
from utils import (
    get_device,
    load_model,
    calculate_metrics,
    print_metrics,
    ensure_dir,
    plot_confusion_matrix,
    compute_majority_baseline,
    compute_xgboost_baseline,
    print_full_comparison,
)


DATA_PATH = "../data/US_Accidents_March23.csv"
OUTPUT_DIR = "../results"
MODEL_TYPE = "mlp"
MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_model.pth")
USE_SELECTED_FEATURES = False  # 默认使用全特征，SHAP 仅作训练后解释
TOP_FEATURES = 15
SHAP_MODEL_TYPE = "rf"


def main():
    print("=" * 70)
    print("加载已训练模型，在测试集上评估")
    print("=" * 70)
    print(f"数据路径       : {DATA_PATH}")
    print(f"模型类型       : {MODEL_TYPE}")
    print(f"模型权重路径   : {MODEL_PATH}")
    print(f"使用 SHAP 特征 : {USE_SELECTED_FEATURES}")
    print("=" * 70)

    device = get_device()
    ensure_dir(OUTPUT_DIR)

    print("\n[1/4] 数据预处理与划分")
    preprocessor = DataPreprocessor(DATA_PATH)
    data = preprocessor.preprocess()
    X_test, y_test = data["X_test"], data["y_test"]

    print("\n[2/5] 基线评估")
    majority_bl = compute_majority_baseline(y_test)
    print(f"  多数类基线: {majority_bl['strategy']}")
    print(f"    占比: {majority_bl['majority_pct']:.2%}, "
          f"Acc: {majority_bl['accuracy']:.4f}, Macro-F1: {majority_bl['f1_macro']:.4f}")

    print("  训练 XGBoost 基线...")
    # Use training data (first split) for XGBoost
    xgb_bl = compute_xgboost_baseline(data["X_train"], data["y_train"], X_test, y_test)
    if xgb_bl:
        print(f"  XGBoost 基线: {xgb_bl['method']}")
        print(f"    Acc: {xgb_bl['accuracy']:.4f}, Macro-F1: {xgb_bl['f1_macro']:.4f}")

    if USE_SELECTED_FEATURES:
        print("\n[3/5] 运行 SHAP 分析以获取相同的特征子集")
        shap_results = run_shap_analysis(
            data, top_n=TOP_FEATURES, model_type=SHAP_MODEL_TYPE, output_dir=OUTPUT_DIR
        )
        feature_indices = shap_results["feature_indices"]
        X_test = X_test[:, feature_indices]
        print(f"使用 SHAP 选择的特征数量: {len(feature_indices)}")
    else:
        print("\n[3/5] 跳过 SHAP，使用全部特征")
        feature_indices = None

    print("\n[4/5] 构建模型并加载已训练权重")
    config = get_default_config()
    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))
    print(f"输入维度: {input_dim}, 类别数: {num_classes}")

    model = create_model(MODEL_TYPE, input_dim, num_classes, config)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型权重文件: {MODEL_PATH}\n"
            f"请先运行训练脚本: cd src && python train.py --model_type {MODEL_TYPE} --use_selected_features"
        )

    model = load_model(model, MODEL_PATH, device)
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
    y_pred = y_pred.cpu().numpy()

    # Model metrics
    model_metrics = calculate_metrics(y_test, y_pred)
    print_metrics(model_metrics, "模型 — 测试集性能")

    # Per-class breakdown
    from utils import print_classification_report
    class_names = [f"Severity {i + 1}" for i in range(num_classes)]
    print_classification_report(y_test, y_pred, class_names)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"  True {class_names[i]:>10s}: {list(row)}")

    # Confusion matrix plot
    class_names = [f"Severity {i + 1}" for i in range(num_classes)]
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_confusion_matrix.png")
    )

    # Multi-baseline comparison
    baselines = {"Majority": majority_bl}
    if xgb_bl:
        baselines["XGBoost"] = xgb_bl
    print_full_comparison(model_metrics, baselines)

    gain = model_metrics["f1_macro"] - max(
        majority_bl["f1_macro"],
        xgb_bl["f1_macro"] if xgb_bl else 0,
    )
    if gain <= 0:
        print("\n⚠  WARNING: 模型 macro-F1 未超过所有基线。")
        print("  深度学习模型在此数据集上可能不必要。")
        print("  建议: 优先使用 XGBoost/CatBoost，或添加 Focal Loss 提升 minority class recall。")

    print("\n[5/5] SHAP 后置解释 (Post-hoc Explanation)")
    print("  提示: 使用 SHAP 在已训练模型上解释预测，而非预筛选特征。")
    print(f"  运行: python task1_shap_analysis.py 进行完整 SHAP 分析")
    print(f"  特征重要性图: {os.path.join(OUTPUT_DIR, 'feature_importance.png')}")

    print("\n评估完成。")


if __name__ == "__main__":
    main()
