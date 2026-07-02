import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import os
import pickle
from typing import Any, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    
    plt.show()


def plot_feature_importance_shap(feature_names: List[str], importance_scores: List[float],
                                 top_n: int = 15, save_path: str = None):
    """
    基于 SHAP 结果的通用特征重要性柱状图。
    与 task1_shap_analysis.py 搭配使用。
    """
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {save_path}")
    
    plt.show()
    
    return df


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()


def save_model(model: torch.nn.Module, save_path: str):
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")


def load_model(model: torch.nn.Module, load_path: str, device: torch.device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    print(f"模型已从 {load_path} 加载")
    return model


def save_data(data: Dict, save_path: str):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def save_json(data: Any, save_path: str) -> str:
    import json
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return save_path


def load_json(load_path: str) -> Any:
    import json
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    print(f"数据已保存至: {save_path}")


def load_data(load_path: str) -> Dict:
    with open(load_path, 'rb') as f:
        data = pickle.load(f)# 持续化加载数据
    print(f"数据已从 {load_path} 加载")
    return data


def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录已创建: {directory}")


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance"):
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print(f"{'='*50}\n")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: List[str] = None):
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def compute_majority_baseline(y_true: np.ndarray, num_classes: int = None) -> Dict[str, float]:
    """计算"始终预测多数类"基线的各项指标。"""
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    counts = np.bincount(y_true, minlength=num_classes)
    majority_class = int(np.argmax(counts))
    y_pred = np.full_like(y_true, majority_class)
    metrics = calculate_metrics(y_true, y_pred)
    metrics["majority_class"] = float(majority_class)
    metrics["majority_pct"] = float(counts[majority_class] / len(y_true))
    return metrics


def compute_xgboost_baseline(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              n_estimators: int = 200, max_depth: int = 8) -> Dict[str, float]:
    """使用 XGBoost 作为树模型基线，在不平衡表格数据上通常强于简单 MLP。"""
    try:
        import xgboost as xgb
    except ImportError:
        print("⚠ XGBoost not installed. Install with: pip install xgboost")
        return None

    num_classes = len(np.unique(y_train))
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    metrics["method"] = f"XGBoost (n={n_estimators}, depth={max_depth})"
    return metrics


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                 random_state: int = 42) -> tuple:
    """对训练集少数类进行 SMOTE 过采样。

    注意: SMOTE 应仅应用于训练集，验证集和测试集保持原始分布。
    返回过采样后的 (X_train, y_train)。
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("⚠ imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X_train, y_train

    counts = np.bincount(y_train)
    if max(counts) / min(counts) < 2.0:
        print("  类别分布较均衡，跳过 SMOTE")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(counts) - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)
    new_counts = np.bincount(y_res)
    print(f"  SMOTE: {counts.tolist()} → {new_counts.tolist()} (增加 {len(y_res) - len(y_train)} 样本)")
    return X_res, y_res


def print_full_comparison(model_metrics: Dict[str, float],
                           baselines: Dict[str, Dict[str, float]]):
    """打印模型 vs 多个基线的完整对比表。"""
    print("\n" + "=" * 80)
    print("模型 vs 基线 — 完整对比")
    print("=" * 80)
    for name, bl in baselines.items():
        desc = bl.get("method", bl.get("strategy", name))
        print(f"  {name}: {desc}")
    print()
    header = f"{'指标':<22s}"
    for name in baselines:
        header += f" {name:>14s}"
    header += f" {'模型':>14s}"
    print(header)
    print("-" * (22 + 16 * (len(baselines) + 1)))

    metric_keys = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 (macro)"),
        ("f1_weighted", "F1 (weighted)"),
        ("recall_macro", "Recall (macro)"),
        ("precision_macro", "Precision (macro)"),
    ]
    for key, label in metric_keys:
        row = f"{label:<22s}"
        best_baseline = -1.0
        for name in baselines:
            val = baselines[name].get(key, 0)
            row += f" {val:>14.4f}"
            if val > best_baseline:
                best_baseline = val
        model_val = model_metrics.get(key, 0)
        delta = model_val - best_baseline
        symbol = "✓" if delta > 0 else "✗"
        row += f" {model_val:>14.4f} {symbol}"
        print(row)

    print("-" * (22 + 16 * (len(baselines) + 1)))
    print(f"  ✓ = 超过最佳基线    ✗ = 未超过最佳基线")
    print("=" * 80)


def plot_feature_importance(feature_importance: np.ndarray,
                            feature_names: List[str],
                            save_path: str = None,
                            top_n: int = 15):
    """
    绘制特征重要性图（基于FT-Transformer的Attention）
    
    Args:
        feature_importance: 特征重要性 [batch_size, n_features]
        feature_names: 特征名称列表
        save_path: 保存路径
        top_n: 显示的Top特征数量
    """
    # 计算平均重要性
    avg_importance = feature_importance.mean(axis=0)
    
    # 选择Top特征
    top_indices = np.argsort(avg_importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_values = avg_importance[top_indices]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = plt.barh(range(top_n), top_values, color=colors)
    
    # 设置y轴标签
    plt.yticks(range(top_n), top_features)
    plt.xlabel('特征重要性 (Attention权重)', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.title('FT-Transformer 特征重要性 (基于Attention)', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=10)
    
    # 添加网格
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {save_path}")
    
    plt.show()


def plot_attention_heatmap(attention_weights: np.ndarray,
                        feature_names: List[str],
                        sample_indices: List[int] = None,
                        save_path: str = None):
    """
    绘制注意力热力图
    
    Args:
        attention_weights: 注意力权重 [batch_size, n_features]
        feature_names: 特征名称列表
        sample_indices: 要显示的样本索引
        save_path: 保存路径
    """
    # 如果没有指定样本，选择前几个
    if sample_indices is None:
        n_samples = min(4, attention_weights.shape[0])
        sample_indices = list(range(n_samples))
    
    # 选择样本
    selected_attention = attention_weights[sample_indices]
    
    # 创建图表
    fig, axes = plt.subplots(len(sample_indices), 1, 
                           figsize=(14, 3 * len(sample_indices)))
    
    for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
        im = ax.imshow(selected_attention[idx:idx+1, :].T, 
                     cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
        ax.set_yticks([])
        ax.set_title(f'样本 {sample_idx} 的注意力分布', fontsize=11)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('注意力权重', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力热力图已保存至: {save_path}")
    
    plt.show()
