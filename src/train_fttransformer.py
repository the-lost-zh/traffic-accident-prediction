#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FT-Transformer模型训练脚本
专为Linux系统优化，支持GPU加速
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from task1_shap_analysis import run_shap_analysis
from task3_classifier import train_classifier, get_default_config, create_model
from utils import set_seed, get_device, ensure_dir, plot_feature_importance, plot_attention_heatmap
from agent import PredictiveAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FT-Transformer模型训练脚本')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/US_Accidents_March23.csv',
                        help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    
    # SHAP分析参数
    parser.add_argument('--skip_shap', action='store_true',
                        help='跳过SHAP分析，直接使用全部特征')
    parser.add_argument('--top_features', type=int, default=15,
                        help='SHAP选择的Top特征数量')
    parser.add_argument('--shap_model_type', type=str, default='rf', choices=['rf', 'nn'],
                        help='SHAP分析使用的模型类型')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='fttransformer',
                        choices=['linear', 'mlp', 'transformer', 'fttransformer'],
                        help='分类器模型类型')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='学习率')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    
    # FT-Transformer特有的参数
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer模型维度')
    parser.add_argument('--nhead', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Transformer层数')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='前馈网络维度')
    
    # 其他参数
    parser.add_argument('--use_selected_features', action='store_true',
                        help='使用SHAP选择的特征进行训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default=None,
                        help='指定设备 (cuda/cpu)')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 选择显卡（0-7），通过 CUDA_VISIBLE_DEVICES 控制
    # 用法示例：
    #   python train_fttransformer.py --device 0   # 使用第 0 号 GPU
    #   python train_fttransformer.py --device 3   # 使用第 3 号 GPU
    if args.device is not None:
        if args.device.isdigit():
            gpu_idx = int(args.device)
            if 0 <= gpu_idx <= 7:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print(f"已将 CUDA_VISIBLE_DEVICES 设置为: {gpu_idx}（使用第 {gpu_idx} 号显卡）")
            else:
                print(f"警告: --device 需在 0~7 范围内，当前值为 {gpu_idx}，将忽略该设置。")
        else:
            print(f"警告: 当前仅支持通过 --device 传入 0~7 的显卡编号，收到参数: {args.device}，将忽略该设置。")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 打印开始信息
    print(f"\n=== 开始训练 FT-Transformer 模型 ===")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型类型: {args.model_type}")
    
    # 1. 数据预处理
    print("\n1. 数据预处理...")
    preprocessor = DataPreprocessor(args.data_path)
    try:
        data_dict = preprocessor.preprocess()
        X = data_dict['X_train']
        y = data_dict['y_train']
        feature_names = data_dict['feature_names']
        print(f"✓ 数据预处理完成: 特征数量 = {X.shape[1]}, 训练/验证/测试 = {len(X)}/{len(data_dict['X_val'])}/{len(data_dict['X_test'])}")
    except Exception as e:
        print(f"✗ 数据预处理失败: {e}")
        return
    
    # 2. SHAP特征重要性分析
    selected_features = None
    if not args.skip_shap:
        print("\n2. SHAP特征重要性分析...")
        try:
            # 使用run_shap_analysis函数
            shap_results = run_shap_analysis(
                data_dict,
                top_n=args.top_features,
                model_type=args.shap_model_type,
                output_dir=args.output_dir
            )
            
            selected_features = shap_results['selected_features']
            print(f"✓ SHAP分析完成: 选择了 {len(selected_features)} 个关键特征")
            print(f"  关键特征: {selected_features[:5]}...")
            
        except Exception as e:
            print(f"✗ SHAP分析失败: {e}")
            print("  继续使用全部特征...")
    
    # 3. 特征选择
    if args.use_selected_features and selected_features:
        print("\n3. 使用选择的特征...")
        try:
            feature_indices = [feature_names.index(f) for f in selected_features]
            X_selected = data_dict['X_train'][:, feature_indices]
            X_val_selected = data_dict['X_val'][:, feature_indices]
            X_test_selected = data_dict['X_test'][:, feature_indices]
            print(f"✓ 特征选择完成: 从 {data_dict['X_train'].shape[1]} 个特征中选择了 {X_selected.shape[1]} 个")
            data_dict['X_train'] = X_selected
            data_dict['X_val'] = X_val_selected
            data_dict['X_test'] = X_test_selected
            feature_names = selected_features
            preprocessor.feature_names = selected_features
        except Exception as e:
            print(f"✗ 特征选择失败: {e}")
            print("  继续使用全部特征...")
    
    # 4. 模型训练
    print("\n4. 模型训练...")
    
    # 配置参数
    config = get_default_config()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward
    })
    
    # 打印配置信息
    print("\n配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # 调用训练函数（验证集早停，测试集最终评估）
        results = train_classifier(
            data_dict['X_train'], data_dict['y_train'],
            data_dict['X_val'], data_dict['y_val'],
            data_dict['X_test'], data_dict['y_test'],
            model_type=args.model_type,
            config=config,
            output_dir=args.output_dir,
            use_class_weights=True
        )
        
        print("\n✓ 模型训练完成!")
        print(f"  训练准确率: {results['train_metrics']['accuracy']:.4f}")
        print(f"  测试准确率: {results['test_metrics']['accuracy']:.4f}")
        
        # 特征重要性可视化（仅对FT-Transformer有效）
        if args.model_type == 'fttransformer':
            print("\n5. 特征重要性可视化...")
            try:
                # 加载训练好的模型
                from task3_classifier import create_model
                model = create_model('fttransformer', data_dict['X_train'].shape[1], 4, config)
                model_path = os.path.join(args.output_dir, 'fttransformer_model.pth')
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=get_device()))
                    print(f"✓ 模型已加载: {model_path}")
                else:
                    print(f"✗ 模型文件不存在: {model_path}")
                    model = None
                
                if model is not None:
                    feat_names = data_dict.get('feature_names', feature_names)
                    sample_X = torch.FloatTensor(data_dict['X_train'][:100]).to(get_device())
                    feature_importance = model.get_feature_importance(sample_X)
                    feature_importance_np = feature_importance.cpu().numpy()
                    importance_path = os.path.join(args.output_dir, 'fttransformer_feature_importance.png')
                    plot_feature_importance(
                        feature_importance_np,
                        feat_names,
                        save_path=importance_path,
                        top_n=20
                    )
                    
                    heatmap_path = os.path.join(args.output_dir, 'fttransformer_attention_heatmap.png')
                    plot_attention_heatmap(
                        feature_importance_np,
                        feat_names,
                        sample_indices=[0, 1, 2, 3],
                        save_path=heatmap_path
                    )
                    
                    print(f"✓ 特征重要性可视化完成!")
                    
            except Exception as e:
                print(f"✗ 特征重要性可视化失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存最终结果
        final_result_path = os.path.join(args.output_dir, 'final_results.json')
        with open(final_result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 最终结果保存到: {final_result_path}")
        
        # 初始化可解释智能体并保存
        print("\n6. 初始化可解释智能体并保存...")
        try:
            preprocessor.save(os.path.join(args.output_dir, 'preprocessor.pkl'))
            agent = PredictiveAgent(model_dir=args.output_dir, device=get_device())
            model = create_model(args.model_type, data_dict['X_train'].shape[1], 4, config)
            model_path = os.path.join(args.output_dir, f'{args.model_type}_model.pth')
            model.load_state_dict(torch.load(model_path, map_location=get_device()))
            agent.model = model
            agent.preprocessor = preprocessor
            bg_data = data_dict['X_train'][:100]
            agent.setup_explainer(bg_data)
            print(f"✓ 智能体解释器（SHAP）保存成功")
        except Exception as e:
            print(f"✗ 智能体初始化或保存失败: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n✗ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n=== 训练完成 ===")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
