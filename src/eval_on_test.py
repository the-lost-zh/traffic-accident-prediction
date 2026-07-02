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
)


DATA_PATH = "../data/US_Accidents_March23.csv"
OUTPUT_DIR = "../results"
MODEL_TYPE = "mlp"
MODEL_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_model.pth")
USE_SELECTED_FEATURES = True
TOP_FEATURES = 15
SHAP_MODEL_TYPE = "rf"


def majority_baseline_metrics(y_true, num_classes=4):
    """Compute metrics if model always predicts the majority class."""
    counts = np.bincount(y_true, minlength=num_classes)
    majority_class = int(np.argmax(counts))
    y_pred = np.full_like(y_true, majority_class)
    return {
        "strategy": f"Always predict Severity {majority_class + 1}",
        "majority_pct": float(counts[majority_class] / len(y_true)),
        **calculate_metrics(y_true, y_pred),
    }


def print_per_class_metrics(y_true, y_pred, num_classes=4):
    class_names = [f"Severity {i + 1}" for i in range(num_classes)]
    print("\n" + "=" * 60)
    print("Per-Class Metrics")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    header = "         " + "".join(f"Pred {c:>8s}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"True {class_names[i]:>8s}  " + "".join(f"{v:>13d}" for v in row))


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

    print("\n[2/4] 多数类基线评估 (Majority-Class Baseline)")
    baseline = majority_baseline_metrics(y_test)
    print(f"  策略: {baseline['strategy']}")
    print(f"  多数类占比: {baseline['majority_pct']:.4f} ({baseline['majority_pct'] * 100:.1f}%)")
    print(f"  基线 Accuracy: {baseline['accuracy']:.4f}")
    print(f"  基线 Macro-F1:  {baseline['f1_macro']:.4f}")
    print(f"  基线 Weighted-F1: {baseline['f1_weighted']:.4f}")

    if USE_SELECTED_FEATURES:
        print("\n[3/4] 运行 SHAP 分析以获取相同的特征子集")
        shap_results = run_shap_analysis(
            data, top_n=TOP_FEATURES, model_type=SHAP_MODEL_TYPE, output_dir=OUTPUT_DIR
        )
        feature_indices = shap_results["feature_indices"]
        X_test = X_test[:, feature_indices]
        print(f"使用 SHAP 选择的特征数量: {len(feature_indices)}")
    else:
        print("\n[3/4] 跳过 SHAP，使用全部特征")
        feature_indices = None

    print("\n[4/4] 构建模型并加载已训练权重")
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
    print_per_class_metrics(y_test, y_pred, num_classes=num_classes)

    # Confusion matrix plot
    class_names = [f"Severity {i + 1}" for i in range(num_classes)]
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_confusion_matrix.png")
    )

    # Comparison summary
    print("\n" + "=" * 60)
    print("模型 vs 多数类基线 — 对比")
    print("=" * 60)
    print(f"{'指标':<22s} {'基线 (全预测多数类)':>22s} {'模型':>22s} {'提升':>10s}")
    print("-" * 76)
    for key, label in [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 (macro)"),
        ("f1_weighted", "F1 (weighted)"),
        ("recall_macro", "Recall (macro)"),
        ("precision_macro", "Precision (macro)"),
    ]:
        base_val = baseline[key]
        model_val = model_metrics[key]
        delta = model_val - base_val
        print(f"{label:<22s} {base_val:>22.4f} {model_val:>22.4f} {delta:>+10.4f}")

    gain = model_metrics["f1_macro"] - baseline["f1_macro"]
    if gain <= 0:
        print("\n⚠  WARNING: 模型 macro-F1 未超过多数类基线。")
        print("  在严重类别不平衡的情况下，accuracy 没有参考价值。")
        print("  建议: 增大 class weights、使用 Focal Loss、或对少数类过采样。")

    print("\n评估完成。")


if __name__ == "__main__":
    main()
