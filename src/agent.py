import os
import joblib
import torch
import numpy as np
import pandas as pd
import shap
from typing import Dict, Tuple, Any

# Ensure we can import from src
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from task3_classifier import create_model, get_default_config
from utils import get_device

class PredictiveAgent:
    """
    交互式预测智能体核心类，封装了数据预处理、模型预测以及SHAP特征解释。
    适合在 API 或实时终端中使用。
    """
    def __init__(self, model_dir: str = 'models', device=None):
        self.model_dir = model_dir
        self.device = device if device else get_device()
        self.preprocessor = None
        self.model = None
        self.explainer = None
        self.config = get_default_config()
        self.is_loaded = False
        
    def load(self, model_type: str = 'mlp', input_dim: int = 10, num_classes: int = 4):
        """加载已保存的预处理器和模型权重"""
        try:
            # 1. 加载预处理器
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = DataPreprocessor.load(preprocessor_path)
            else:
                print(f"未能找到预处理器文件: {preprocessor_path}，请确保先完成训练。")
                return False
                
            # 2. 加载模型
            model_path = os.path.join(self.model_dir, f'{model_type}_model.pth')
            # 使用预处理器的特征维度如果有的话
            if hasattr(self.preprocessor, 'feature_names') and len(self.preprocessor.feature_names) > 0:
                actual_input_dim = len(self.preprocessor.feature_names)
            else:
                actual_input_dim = input_dim
                
            self.model = create_model(model_type, actual_input_dim, num_classes, self.config)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ 模型文件已加载: {model_path}")
            else:
                print(f"未能找到模型文件: {model_path}")
                return False
                
            # 3. 加载 Explainer（如果存在）
            explainer_path = os.path.join(self.model_dir, 'explainer.pkl')
            if os.path.exists(explainer_path):
                self.explainer = joblib.load(explainer_path)
            else:
                print("未找到SHAP Explainer，部分可解释性功能受限。")
                
            self.is_loaded = True
            print(f"✓ 智能体 (Agent) 加载成功! [设备: {self.device}]")
            return True
        except Exception as e:
            print(f"✗ 智能体加载失败: {str(e)}")
            return False
            
    def _predict_proba_wrapper(self, x_numpy):
        """为 SHAP 提供的模型预测包装器"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_numpy).to(self.device)
            outputs = self.model(x_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
        return proba

    def setup_explainer(self, background_data: np.ndarray):
        """使用背景数据初始化SHAP Explainer并保存"""
        print("正在初始化SHAP Explainer...")
        self.explainer = shap.explainers.Permutation(self._predict_proba_wrapper, background_data, seed=42)
        
        # 保存 Explainer
        explainer_path = os.path.join(self.model_dir, 'explainer.pkl')
        joblib.dump(self.explainer, explainer_path)
        print(f"Explainer已保存至 {explainer_path}")

    def preprocess_input(self, data_dict: Dict) -> Tuple[np.ndarray, list]:
        """将原始输入的字典数据转换为模型可接受的Tensor特征，并返回特征名顺序"""
        if not self.preprocessor:
            raise ValueError("预处理器未加载")

        # 转为DataFrame单行
        df = pd.DataFrame([data_dict])
        
        # 分类特征编码
        for col in self.preprocessor.categorical_features:
            if col in df.columns and col in self.preprocessor.label_encoders:
                # 处理未见过的类别
                val = str(df[col].iloc[0])
                le = self.preprocessor.label_encoders[col]
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    # 如果是未知类别，默认取第一个或者取众数。这里简单用 0
                    df[col] = 0
                    
        # 数值特征标准化
        if self.preprocessor.numeric_features:
            # 确保列顺序
            num_cols = [c for c in self.preprocessor.numeric_features if c in df.columns]
            if num_cols:
                df[num_cols] = self.preprocessor.scaler.transform(df[num_cols])
                
        # 确保完整的特征名对应上
        feature_names = self.preprocessor.feature_names
        missing_cols = set(feature_names) - set(df.columns)
        for col in missing_cols:
            df[col] = 0.0 # 填补缺失值
            
        # 按照训练时的顺序排列
        features = df[feature_names].values.astype(np.float32)
        return features, feature_names

    def predict_with_explanation(self, data_dict: Dict) -> Dict[str, Any]:
        """进行预测并生成可解释性结果"""
        if not self.is_loaded:
            raise RuntimeError("智能体未准备好。请先调用 load()。")
            
        # 1. 预处理数据
        features, feature_names = self.preprocess_input(data_dict)
        
        # 2. 模型预测
        features_tensor = torch.FloatTensor(features).to(self.device)
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            max_prob = float(probabilities[predicted_class])
            
        result = {
            'severity': predicted_class,
            'probability': max_prob,
            'probabilities': probabilities.tolist(),
            'feature_contributions': {}
        }
        
        # 3. 解释结果 (如果 Explainer 已加载)
        if self.explainer is not None:
            try:
                # max_evals 计算量可以适当调小以加速实时推理
                n_features = features.shape[1]
                evals = max(2, 500 // (n_features + 1)) * (n_features + 1)
                shap_out = self.explainer(features, max_evals=evals)
                
                vals = shap_out.values if hasattr(shap_out, 'values') else shap_out
                # 统一为 ndarray
                if isinstance(vals, list):
                    vals = np.stack(vals, axis=0) if all(hasattr(a, 'shape') for a in vals) else np.array(vals)
                else:
                    vals = np.asarray(vals)
                    
                # 针对当前预测类别的特征重要性 [samples, features]
                if vals.ndim == 3:
                     if vals.shape[-1] < vals.shape[1]:
                          # (1, features, classes) -> (features,)
                          contributions = vals[0, :, predicted_class]
                     else:
                          # (classes, 1, features) -> (features,)
                          contributions = vals[predicted_class, 0, :]
                else:
                    contributions = vals[0]
                    
                # 构建贡献度字典
                for name, contrib in zip(feature_names, contributions):
                    result['feature_contributions'][name] = float(contrib)
                    
            except Exception as e:
                print(f"SHAP 解释过程出错: {e}")
                
        return result
