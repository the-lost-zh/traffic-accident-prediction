import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("测试SHAP分析")
print("="*50)

from data_preprocessing import DataPreprocessor
from task1_shap_analysis import SHAPAnalyzer

preprocessor = DataPreprocessor('../data/US_Accidents_March23.csv')
data_dict = preprocessor.preprocess()

print("\n创建SHAP分析器...")
analyzer = SHAPAnalyzer(
    X_train=data_dict['X_train'][:1000],
    y_train=data_dict['y_train'][:1000],
    X_test=data_dict['X_test'][:200],
    y_test=data_dict['y_test'][:200],
    feature_names=data_dict['feature_names']
)

print("\n训练随机森林模型...")
analyzer.train_base_model(model_type='rf')

print("\n计算SHAP值...")
try:
    feature_importance = analyzer.compute_shap_values(model_type='rf', sample_size=50)
    print("✓ SHAP值计算成功")
    print(f"特征重要性形状: {feature_importance.shape}")
    print("Top 5 特征:")
    print(feature_importance.head())
except Exception as e:
    print(f"✗ SHAP值计算失败: {e}")
    import traceback
    traceback.print_exc()

print("="*50)
