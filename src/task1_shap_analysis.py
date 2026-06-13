import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable, Any
import os
import sys

# 兼容两种运行方式：
# 1) 在项目根目录运行: PYTHONPATH=. python src/train_fttransformer.py
# 2) 在 src 目录运行:   python train_fttransformer.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # 方式1：从项目根目录看，src 是包
    from src.utils import (
        set_seed,
        get_device,
        plot_feature_importance_shap,
        save_data,
        ensure_dir,
    )
except ModuleNotFoundError:
    # 方式2：在 src 目录运行，直接相对导入
    from utils import (
        set_seed,
        get_device,
        plot_feature_importance_shap,
        save_data,
        ensure_dir,
    )


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 4):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class SHAPAnalyzer:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str], device: torch.device = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.device = device if device else get_device()
        self.model = None
        self.shap_values = None
        self.feature_importance = None
        
    def train_base_model(self, model_type: str = 'rf', epochs: int = 50, batch_size: int = 512):
        print(f"\n=== 训练基础模型 ({model_type.upper()}) ===")
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
            
            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)
            
            print(f"训练集准确率: {accuracy_score(self.y_train, train_pred):.4f}")
            print(f"测试集准确率: {accuracy_score(self.y_test, test_pred):.4f}")
            
        elif model_type == 'nn':
            input_dim = self.X_train.shape[1]
            num_classes = len(np.unique(self.y_train))
            
            self.model = SimpleNN(input_dim, num_classes=num_classes).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
            y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_test_tensor = torch.LongTensor(self.y_test).to(self.device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        train_pred = self.model(X_train_tensor).argmax(dim=1).cpu().numpy()
                        test_pred = self.model(X_test_tensor).argmax(dim=1).cpu().numpy()
                        train_acc = accuracy_score(self.y_train, train_pred)
                        test_acc = accuracy_score(self.y_test, test_pred)
                    
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, "
                          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        print("基础模型训练完成")
    
    def _predict_proba_wrapper(self, X: np.ndarray) -> np.ndarray:
        """NN 预测概率包装器，供 PermutationExplainer 使用。"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def compute_shap_values(self, model_type: str = 'rf', sample_size: int = 100,
                            use_permutation: bool = True, max_evals: int = 500):
        """
        计算 SHAP 值。推荐：表格数据使用 PermutationExplainer（最新最佳实践）；
        树模型可选 TreeExplainer（精确且快）。
        """
        print(f"\n=== 计算SHAP值 (use_permutation={use_permutation}) ===")
        X_explain = self.X_test[:sample_size]
        n_features = X_explain.shape[1]
        background_size = min(100, len(self.X_train))

        if model_type == 'rf' and not use_permutation:
            # TreeExplainer：精确、快速，仅适用于树模型
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_explain)
        else:
            # PermutationExplainer：模型无关、保证局部准确性，推荐用于表格数据
            background = self.X_train[:background_size]
            if model_type == 'rf':
                model_fn: Callable = lambda x: self.model.predict_proba(x)
            else:
                model_fn = self._predict_proba_wrapper
            explainer = shap.explainers.Permutation(model_fn, background, seed=42)
            # 新 API：__call__ 返回 Explanation；max_evals 控制精度与速度
            evals_per_feature = max(2, max_evals // (n_features + 1))
            try:
                shap_out = explainer(X_explain, max_evals=evals_per_feature * (n_features + 1))
            except TypeError:
                shap_out = explainer(X_explain)
            if hasattr(shap_out, 'values'):
                vals = shap_out.values
            else:
                vals = shap_out
            # PermutationExplainer 多分类时 values 可能是 list of arrays，需统一为 ndarray
            if isinstance(vals, list):
                vals = np.stack(vals, axis=0) if all(hasattr(a, 'shape') for a in vals) else np.array(vals)
            else:
                vals = np.asarray(vals)
            # 多分类时 3D：可能是 (n_classes, n_samples, n_features) 或 (n_samples, n_features, n_classes)
            # 目标得到 (n_samples, n_features) 供 summary_plot 使用
            if vals.ndim == 3:
                if vals.shape[-1] < vals.shape[1]:
                    # (n_samples, n_features, n_classes) -> 对类别维取均值得 (n_samples, n_features)
                    self.shap_values = np.abs(vals).mean(axis=-1)
                else:
                    # (n_classes, n_samples, n_features) -> 对类别维取均值得 (n_samples, n_features)
                    self.shap_values = np.abs(vals).mean(axis=0)
            else:
                self.shap_values = vals

        # TreeExplainer 多分类返回 list of arrays，统一转为 (n_classes, n_samples, n_features) 再聚合
        if isinstance(self.shap_values, list):
            self.shap_values = np.stack(self.shap_values, axis=0) if all(
                hasattr(a, 'shape') for a in self.shap_values
            ) else np.array(self.shap_values)
        self.shap_values = np.asarray(self.shap_values)
        # 聚合为每特征一维：(n_samples, n_features) -> (n_features,)
        if self.shap_values.ndim == 3:
            mean_shap = np.abs(self.shap_values).mean(axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        mean_shap = np.asarray(mean_shap).flatten()
        # 只截断到有效长度，不改变 feature_names 数量（避免把 4 类当成 4 特征）
        n_imp = len(mean_shap)
        n_fnames = len(self.feature_names)
        if n_imp != n_fnames:
            min_length = min(n_fnames, n_imp)
            self.feature_names = self.feature_names[:min_length]
            mean_shap = mean_shap[:min_length]

        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)

        print("SHAP值计算完成")
        print("Top 10 重要特征:")
        print(self.feature_importance.head(10).to_string(index=False))
        return self.feature_importance
    
    def select_top_features(self, top_n: int = 15) -> Tuple[List[str], List[int]]:
        top_features_df = self.feature_importance.head(top_n)
        selected_features = top_features_df['feature'].tolist()
        
        feature_indices = [self.feature_names.index(feat) for feat in selected_features]
        
        print(f"\n=== 选定的Top {top_n}关键特征 ===")
        for i, (feat, imp) in enumerate(zip(selected_features, top_features_df['importance'].values), 1):
            print(f"{i:2d}. {feat:30s} (重要性: {imp:.4f})")
        
        return selected_features, feature_indices
    
    def plot_shap_summary(self, save_path: str = None, plot_type: str = "dot"):
        """绘制 SHAP 摘要图。plot_type: 'dot' 或 'bar'。要求 sv 为 (n_samples, n_features)。"""
        print("\n=== 绘制SHAP摘要图 ===")
        sv = self.shap_values
        if isinstance(sv, list):
            sv = np.stack(sv, axis=0) if all(hasattr(a, 'shape') for a in sv) else np.array(sv)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            if sv.shape[-1] < sv.shape[1]:
                sv = np.abs(sv).mean(axis=-1)
            else:
                sv = np.abs(sv).mean(axis=0)
        # summary_plot 要求 shap_values 与 data 均为 (n_samples, n_features)，且样本数、特征数一致
        n_sv_samples, n_sv_features = sv.shape[0], sv.shape[1]
        n_plot = min(100, n_sv_samples, len(self.X_test))
        n_feat = min(n_sv_features, self.X_test.shape[1])
        X_plot = self.X_test[:n_plot, :n_feat]
        sv_plot = sv[:n_plot, :n_feat]
        feat_names_plot = self.feature_names[:n_feat]
        plt.figure(figsize=(12, 10))
        shap.summary_plot(sv_plot, X_plot, feature_names=feat_names_plot, show=False, plot_type=plot_type)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP摘要图已保存至: {save_path}")
        plt.close()
        # bar 图单独保存一份便于对比
        if save_path and plot_type == "dot":
            bar_path = save_path.replace(".png", "_bar.png")
            plt.figure(figsize=(12, 10))
            shap.summary_plot(sv_plot, X_plot, feature_names=feat_names_plot, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP条形摘要图已保存至: {bar_path}")
    
    def save_results(self, output_dir: str = 'results'):
        ensure_dir(output_dir)
        
        results = {
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names
        }
        
        save_path = os.path.join(output_dir, 'shap_results.pkl')
        save_data(results, save_path)
        
        csv_path = os.path.join(output_dir, 'feature_importance.csv')
        self.feature_importance.to_csv(csv_path, index=False)
        print(f"特征重要性已保存至: {csv_path}")


def run_shap_analysis(data_dict: Dict, top_n: int = 15,
                      model_type: str = 'rf', output_dir: str = 'results',
                      use_permutation: bool = True, shap_sample_size: int = 100,
                      max_evals: int = 500):
    """
    运行 SHAP 特征分析。默认使用 PermutationExplainer（表格数据推荐）。
    use_permutation=False 时，树模型使用 TreeExplainer。
    """
    set_seed(42)
    # 兼容仅有 train/test 的 data_dict（无 X_val 时用 X_test 做解释）
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict.get('X_test', data_dict.get('X_val'))
    y_test = data_dict.get('y_test', data_dict.get('y_val'))
    if X_test is None:
        X_test = X_train[:min(500, len(X_train))]
        y_test = y_train[:min(500, len(y_train))]
    analyzer = SHAPAnalyzer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=data_dict['feature_names']
    )
    analyzer.train_base_model(model_type=model_type)
    analyzer.compute_shap_values(
        model_type=model_type,
        sample_size=shap_sample_size,
        use_permutation=use_permutation,
        max_evals=max_evals
    )
    selected_features, feature_indices = analyzer.select_top_features(top_n=top_n)
    ensure_dir(output_dir)
    plot_feature_importance_shap(
        analyzer.feature_names,
        analyzer.feature_importance['importance'].values,
        top_n=20,
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    analyzer.plot_shap_summary(save_path=os.path.join(output_dir, 'shap_summary.png'))
    analyzer.save_results(output_dir=output_dir)
    return {
        'selected_features': selected_features,
        'feature_indices': feature_indices,
        'feature_importance': analyzer.feature_importance
    }


if __name__ == '__main__':
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor('../data/US_Accidents_March23.csv')
    data_dict = preprocessor.preprocess()
    
    results = run_shap_analysis(data_dict, top_n=15, model_type='rf')
    
    print("ne\n=== SHAP分析完成 ===")
    print(f"选定的关键特征: {len(results['selected_features'])} 个")
