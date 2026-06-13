import os
import sys
import argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from task1_shap_analysis import run_shap_analysis
from task2_gan import train_gan
from task3_classifier import train_classifier, get_default_config
from utils import set_seed, get_device, save_data, ensure_dir
from agent import PredictiveAgent


def parse_args():
    parser = argparse.ArgumentParser(description='交通事故损伤等级预测系统')
    
    parser.add_argument('--data_path', type=str, default='../data/US_Accidents_March23.csv',
                       help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='输出目录')
    parser.add_argument('--skip_shap', action='store_true',
                       help='跳过SHAP分析，直接使用已有特征')
    parser.add_argument('--top_features', type=int, default=15,
                       help='SHAP分析选择的Top特征数量')
    parser.add_argument('--shap_model_type', type=str, default='rf',
                       choices=['rf', 'nn'],
                       help='SHAP分析使用的模型类型')
    parser.add_argument('--model_type', type=str, default='linear',
                       choices=['linear', 'mlp', 'transformer'],
                       help='分类器模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率')
    parser.add_argument('--use_selected_features', action='store_true',
                       help='使用SHAP选择的特征进行训练')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--train_gan', action='store_true',
                       help='训练GAN模型')
    parser.add_argument('--gan_epochs', type=int, default=100,
                       help='GAN训练轮数')
    parser.add_argument('--gan_batch_size', type=int, default=128,
                       help='GAN批次大小')
    parser.add_argument('--gan_latent_dim', type=int, default=100,
                       help='GAN潜在空间维度')
    parser.add_argument('--gan_learning_rate', type=float, default=0.0002,
                       help='GAN学习率')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)
    
    print("="*70)
    print("交通事故损伤等级预测系统")
    print("="*70)
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用设备: {device}")
    print(f"随机种子: {args.seed}")
    print("="*70)
    
    print("\n[步骤 1/4] 数据预处理")
    print("-"*70)
    preprocessor = DataPreprocessor(args.data_path)
    data_dict = preprocessor.preprocess()
    
    print("\n[步骤 2/4] SHAP特征分析")
    print("-"*70)
    
    if args.skip_shap:
        print("跳过SHAP分析，使用全部特征")
        selected_features = data_dict['feature_names']
        feature_indices = list(range(len(selected_features)))
        shap_results = None
    else:
        shap_results = run_shap_analysis(
            data_dict,
            top_n=args.top_features,
            model_type=args.shap_model_type,
            output_dir=args.output_dir
        )
        selected_features = shap_results['selected_features']
        feature_indices = shap_results['feature_indices']
        
        print(f"\nSHAP分析完成，选择了 {len(selected_features)} 个关键特征")
    
    print("\n[步骤 3/4] 模型训练")
    print("-"*70)
    
    config = get_default_config()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['dropout'] = args.dropout
    
    if args.use_selected_features and not args.skip_shap:
        print(f"使用SHAP选择的 {len(selected_features)} 个特征进行训练")
        training_data = {
            'X_train': data_dict['X_train'][:, feature_indices],
            'X_val': data_dict['X_val'][:, feature_indices],
            'X_test': data_dict['X_test'][:, feature_indices],
            'y_train': data_dict['y_train'],
            'y_val': data_dict['y_val'],
            'y_test': data_dict['y_test']
        }
    else:
        print(f"使用全部 {len(data_dict['feature_names'])} 个特征进行训练")
        training_data = {
            'X_train': data_dict['X_train'],
            'X_val': data_dict['X_val'],
            'X_test': data_dict['X_test'],
            'y_train': data_dict['y_train'],
            'y_val': data_dict['y_val'],
            'y_test': data_dict['y_test']
        }

    classifier_results = train_classifier(
        training_data['X_train'],
        training_data['y_train'],
        training_data['X_val'],
        training_data['y_val'],
        training_data['X_test'],
        training_data['y_test'],
        model_type=args.model_type,
        config=config,
        output_dir=args.output_dir
    )
    
    print("\n[步骤 4/4] 训练GAN模型")
    print("-"*70)
    
    gan_results = None
    if args.train_gan:
        gan_config = {
            'epochs': args.gan_epochs,
            'batch_size': args.gan_batch_size,
            'latent_dim': args.gan_latent_dim,
            'learning_rate': args.gan_learning_rate
        }
        
        gan_output_dir = os.path.join(args.output_dir, 'gan_models')
        gan_results = train_gan(
            X_train=training_data['X_train'],
            y_train=training_data['y_train'],
            config=gan_config,
            output_dir=gan_output_dir
        )
    
    print("\n[步骤 5/6] 初始化可解释智能体并保存")
    print("-"*70)
    
    preprocessor.save(os.path.join(args.output_dir, 'preprocessor.pkl'))
    agent = PredictiveAgent(model_dir=args.output_dir, device=device)
    agent.model = classifier_results['model']
    agent.preprocessor = preprocessor
    bg_data = training_data['X_train'][:100]
    agent.setup_explainer(bg_data)
    
    print("\n[步骤 6/6] 保存结果")
    print("-"*70)
    
    final_results = {
        'model_type': args.model_type,
        'config': config,
        'selected_features': selected_features if not args.skip_shap else data_dict['feature_names'],
        'feature_indices': feature_indices if not args.skip_shap else list(range(len(data_dict['feature_names']))),
        'train_metrics': classifier_results['train_metrics'],
        'test_metrics': classifier_results['test_metrics'],
        'history': classifier_results['history'],
        'shap_results': shap_results,
        'gan_results': gan_results
    }
    
    results_path = os.path.join(args.output_dir, 'final_results.json')
    
    def convert_to_serializable(obj):
        """将对象转换为可JSON序列化的格式"""
        if hasattr(obj, 'to_dict'):
            # 处理DataFrame对象
            return obj.to_dict()
        elif isinstance(obj, dict):
            # 递归处理字典
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 处理列表
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            # 处理numpy浮点数
            return float(obj)
        elif isinstance(obj, np.integer):
            # 处理numpy整数
            return int(obj)
        else:
            # 其他类型直接返回
            return obj
    
    serializable_results = convert_to_serializable(final_results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"最终结果已保存至: {results_path}")
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"\n模型类型: {args.model_type.upper()}")
    print(f"测试集准确率: {classifier_results['test_metrics']['accuracy']:.4f}")
    print(f"测试集F1-score (加权): {classifier_results['test_metrics']['f1_weighted']:.4f}")
    print(f"使用的特征数: {len(selected_features)}")
    
    if args.train_gan:
        print(f"\nGAN训练结果:")
        print(f"- 潜在空间维度: {args.gan_latent_dim}")
        print(f"- 训练轮数: {args.gan_epochs}")
        print(f"- 生成的样本数: {gan_results['generated_samples_shape'][0]}")
        print(f"- 模型保存路径: {gan_results['model_path']}")
    
    print("="*70)
    
    return final_results


if __name__ == '__main__':
    results = main()
