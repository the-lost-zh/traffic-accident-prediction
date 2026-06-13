import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.df = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def save(self, filepath: str):
        """保存预处理器的状态以便推理时使用"""
        state = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(state, filepath)
        print(f"数据预处理器已保存至: {filepath}")
        
    @classmethod
    def load(cls, filepath: str):
        """加载保存的预处理器"""
        state = joblib.load(filepath)
        instance = cls()
        instance.numeric_features = state['numeric_features']
        instance.categorical_features = state['categorical_features']
        instance.feature_names = state['feature_names']
        instance.label_encoders = state['label_encoders']
        instance.scaler = state['scaler']
        print(f"数据预处理器已从 {filepath} 加载")
        return instance
        
    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成，共 {len(self.df)} 条记录，{len(self.df.columns)} 个特征")
        return self.df
    
    def analyze_data(self):
        print("\n=== 数据概览 ===")
        print(self.df.info())
        
        print("\n=== 缺失值统计 ===")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing Percentage': missing_pct
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\n=== 目标变量分布 ===")
        print(self.df['Severity'].value_counts().sort_index())
        
        return missing_df
    
    def handle_missing_values(self):
        print("\n=== 处理缺失值 ===")
        
        for col in self.df.columns:
            if col == 'Severity':
                continue
                
            missing_pct = self.df[col].isnull().sum() / len(self.df)
            
            if missing_pct > 0.5:
                print(f"删除缺失率过高的列: {col} ({missing_pct:.2%})")
                self.df = self.df.drop(columns=[col])
            elif self.df[col].dtype in ['int64', 'float64']:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
            else:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
        
        print(f"缺失值处理完成，剩余特征数: {len(self.df.columns)}")
    
    def identify_feature_types(self):
        for col in self.df.columns:
            if col == 'Severity':
                continue
                
            if self.df[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            else:
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.05:
                    self.categorical_features.append(col)
                else:
                    self.df = self.df.drop(columns=[col])
        
        print(f"\n数值型特征: {len(self.numeric_features)} 个")
        print(f"分类型特征: {len(self.categorical_features)} 个")
    
    def encode_categorical_features(self):
        print("\n=== 编码分类特征 ===")
        
        for col in self.categorical_features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"编码特征: {col} (类别数: {len(le.classes_)})")
    
    def scale_numeric_features(self):
        print("\n=== 标准化数值特征 ===")
        
        if self.numeric_features:
            self.df[self.numeric_features] = self.scaler.fit_transform(self.df[self.numeric_features])
            print(f"已标准化 {len(self.numeric_features)} 个数值特征")
    
    def prepare_features_and_target(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        target = self.df['Severity'].values - 1
        features = self.df.drop(columns=['Severity']).values
        self.feature_names = self.df.drop(columns=['Severity']).columns.tolist()
        feature_names = self.feature_names
        
        print(f"\n特征矩阵形状: {features.shape}")
        print(f"目标变量形状: {target.shape}")
        print(f"目标类别数: {len(np.unique(target))}")
        
        return features, target, feature_names
    
    def split_data(self, features: np.ndarray, target: np.ndarray,
                   train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                   random_state: int = 42) -> Tuple:
        """划分训练集、验证集、测试集（分层抽样）。"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和须为1"
        # 先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, test_size=test_ratio, random_state=random_state, stratify=target
        )
        # 再从剩余数据中分出验证集（占剩余的比例 = val_ratio / (1 - test_ratio)）
        val_size = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        print(f"\n数据集划分 (训练/验证/测试):")
        print(f"  训练集: {len(X_train)} 样本 ({100*train_ratio:.0f}%)")
        print(f"  验证集: {len(X_val)} 样本 ({100*val_ratio:.0f}%)")
        print(f"  测试集: {len(X_test)} 样本 ({100*test_ratio:.0f}%)")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                  test_ratio: float = 0.15, random_state: int = 42) -> Dict:
        self.load_data()
        self.analyze_data()
        self.handle_missing_values()
        self.identify_feature_types()
        self.encode_categorical_features()
        self.scale_numeric_features()

        features, target, feature_names = self.prepare_features_and_target()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            features, target,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
            random_state=random_state
        )

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }


if __name__ == '__main__':
    preprocessor = DataPreprocessor('data/US_Accidents_March23.csv')
    data_dict = preprocessor.preprocess()
    
    print("\n=== 预处理完成 ===")
    print(f"可用特征: {len(data_dict['feature_names'])} 个")
