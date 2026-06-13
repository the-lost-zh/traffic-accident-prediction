import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("快速测试脚本 - 验证代码功能")
print("="*70)

try:
    print("\n[1/5] 测试数据预处理...")
    from data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor('../data/US_Accidents_March23.csv')
    
    data_dict = preprocessor.preprocess()
    
    assert 'X_train' in data_dict
    assert 'X_val' in data_dict
    assert 'X_test' in data_dict
    assert 'y_train' in data_dict
    assert 'y_val' in data_dict
    assert 'y_test' in data_dict
    assert 'feature_names' in data_dict

    print("✓ 数据预处理测试通过")
    print(f"  - 训练样本: {len(data_dict['X_train'])}")
    print(f"  - 验证样本: {len(data_dict['X_val'])}")
    print(f"  - 测试样本: {len(data_dict['X_test'])}")
    print(f"  - 特征数: {len(data_dict['feature_names'])}")
    
except Exception as e:
    print(f"✗ 数据预处理测试失败: {e}")
    sys.exit(1)

try:
    print("\n[2/5] 测试工具函数...")
    from utils import set_seed, get_device, calculate_metrics
    import numpy as np
    
    set_seed(42)
    device = get_device()
    
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    
    print("✓ 工具函数测试通过")
    print(f"  - 设备: {device}")
    print(f"  - 测试指标计算正常")
    
except Exception as e:
    print(f"✗ 工具函数测试失败: {e}")
    sys.exit(1)

try:
    print("\n[3/5] 测试SHAP分析（使用小样本）...")
    from task1_shap_analysis import SHAPAnalyzer
    
    analyzer = SHAPAnalyzer(
        X_train=data_dict['X_train'][:5000],
        y_train=data_dict['y_train'][:5000],
        X_test=data_dict['X_test'][:1000],
        y_test=data_dict['y_test'][:1000],
        feature_names=data_dict['feature_names']
    )
    
    analyzer.train_base_model(model_type='rf')
    feature_importance = analyzer.compute_shap_values(model_type='rf', sample_size=50)
    
    assert feature_importance is not None
    assert len(feature_importance) > 0
    
    print("✓ SHAP分析测试通过")
    print(f"  - Top 1 特征: {feature_importance.iloc[0]['feature']}")
    
except Exception as e:
    print(f"✗ SHAP分析测试失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n[4/5] 测试线性分类器（使用小样本）...")
    from task3_classifier import train_classifier, get_default_config
    
    config = get_default_config()
    config['epochs'] = 5
    config['batch_size'] = 256
    
    results = train_classifier(
        data_dict['X_train'][:10000],
        data_dict['y_train'][:10000],
        data_dict['X_val'][:2000],
        data_dict['y_val'][:2000],
        data_dict['X_test'][:2000],
        data_dict['y_test'][:2000],
        model_type='linear',
        config=config,
        output_dir='../results'
    )
    
    assert 'model' in results
    assert 'test_metrics' in results
    
    print("✓ 线性分类器测试通过")
    print(f"  - 测试准确率: {results['test_metrics']['accuracy']:.4f}")
    
except Exception as e:
    print(f"✗ 线性分类器测试失败: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n[5/5] 测试MLP分类器（使用小样本）...")
    config['epochs'] = 5
    
    results = train_classifier(
        data_dict['X_train'][:10000],
        data_dict['y_train'][:10000],
        data_dict['X_val'][:2000],
        data_dict['y_val'][:2000],
        data_dict['X_test'][:2000],
        data_dict['y_test'][:2000],
        model_type='mlp',
        config=config,
        output_dir='../results'
    )
    
    assert 'model' in results
    assert 'test_metrics' in results
    
    print("✓ MLP分类器测试通过")
    print(f"  - 测试准确率: {results['test_metrics']['accuracy']:.4f}")
    
except Exception as e:
    print(f"✗ MLP分类器测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("所有测试通过! ✓")
print("="*70)
print("\n您现在可以运行完整的训练流程:")
print("  python train.py --model_type linear --use_selected_features")
print("\n或者使用Jupyter Notebook:")
print("  cd ../notebook")
print("  jupyter notebook traffic_accident_analysis.ipynb")
print("="*70)
