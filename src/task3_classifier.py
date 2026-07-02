import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import set_seed, get_device, calculate_metrics, plot_confusion_matrix, plot_training_history, save_model, print_metrics, print_classification_report, ensure_dir


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class NumericalFeatureTokenizer(nn.Module):
    """
    Vectorized Feature Tokenizer for numerical features
    为每个数值特征分配独立的权重和偏置，避免Python for循环
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # 为每个特征分配独立的权重和偏置
        # 权重形状: [1, n_features, d_model] 用于广播
        self.weights = nn.Parameter(torch.randn(1, n_features, d_model) / d_model**0.5)
        self.bias = nn.Parameter(torch.zeros(1, n_features, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features, 1]
        # output: [batch, n_features, d_model]
        return x * self.weights + self.bias


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], num_classes: int = 4, dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4, 
                 d_model: int = 64, nhead: int = 4, num_layers: int = 1, 
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super(TransformerClassifier, self).__init__()
        
        # 添加参数兼容性检查
        if d_model % nhead != 0:
            raise ValueError(f"Transformer参数不兼容: d_model={d_model} 必须能被 nhead={nhead} 整除")
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # 直接处理表格数据，不需要序列维度
        x = self.input_projection(x)  # 形状变为 [batch_size, d_model]
        # 为了使用Transformer，添加序列维度但设置为特征维度
        x = x.unsqueeze(1)  # 形状变为 [batch_size, 1, d_model]
        x = self.transformer(x)
        x = x.squeeze(1)  # 形状变回 [batch_size, d_model]
        return self.classifier(x)


class FTTransformerClassifier(nn.Module):
    """
    Feature Tokenizer Transformer (FT-Transformer) for tabular data
    专门为表格数据设计的Transformer变体
    
    核心思想：
    - 将表格的每一列（特征）视为一个独立的Token
    - 序列长度等于特征数量
    - Attention计算特征之间的相互关系
    """
    def __init__(self, input_dim: int, num_classes: int = 4, 
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super(FTTransformerClassifier, self).__init__()
        
        # 添加参数兼容性检查
        if d_model % nhead != 0:
            raise ValueError(f"Transformer参数不兼容: d_model={d_model} 必须能被 nhead={nhead} 整除")
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Feature Tokenizer: 使用Vectorized层为每个特征分配独立权重
        # 避免Python for循环，速度最快
        self.feature_tokenizer = NumericalFeatureTokenizer(input_dim, d_model)
        
        # CLS Token: 用于分类的特殊token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional Encoding: 为每个特征位置添加位置信息
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 保存注意力权重用于可视化
        self.attention_weights = None
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot 初始化提升收敛与准确率"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.5 if 'tokenizer' in name else 1.0)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        # 限制输入范围，防止异常值导致 NaN/Inf
        x = torch.clamp(x, -10.0, 10.0)
        
        # Feature Tokenization: 使用Vectorized层处理所有特征
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        x = self.feature_tokenizer(x)  # [batch_size, input_dim, d_model]
        # LayerNorm 稳定数值，减少 NaN/Inf
        x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        
        # 添加位置编码
        x = x + self.positional_encoding  # [batch_size, input_dim, d_model]
        
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, input_dim + 1, d_model]
        
        # Transformer Encoder: 计算特征之间的相互关系
        x = self.transformer(x)  # [batch_size, input_dim + 1, d_model]
        
        # 使用CLS Token进行分类
        cls_output = x[:, 0, :]  # [batch_size, d_model]
        
        # Classification
        return self.classifier(cls_output)
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征重要性（基于CLS Token的注意力）
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            feature_importance: 特征重要性 [batch_size, input_dim]
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)

            # Feature Tokenization
            x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
            x = self.feature_tokenizer(x)  # [batch_size, input_dim, d_model]
            
            # 添加位置编码
            x = x + self.positional_encoding  # [batch_size, input_dim, d_model]
            
            # 添加CLS Token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
            x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [batch_size, input_dim + 1, d_model]
            
            # Transformer Encoder
            x_encoded = self.transformer(x_with_cls)  # [batch_size, input_dim + 1, d_model]
            
            # 计算特征重要性：CLS Token与每个特征token的相似度
            cls_output = x_encoded[:, 0, :]  # [batch_size, d_model]
            feature_outputs = x_encoded[:, 1:, :]  # [batch_size, input_dim, d_model]
            
            # 使用余弦相似度计算重要性
            cls_normalized = cls_output / (cls_output.norm(dim=1, keepdim=True) + 1e-8)
            feature_normalized = feature_outputs / (feature_outputs.norm(dim=2, keepdim=True) + 1e-8)
            
            # 计算相似度
            feature_importance = torch.sum(cls_normalized.unsqueeze(1) * feature_normalized, dim=2)
            
            return feature_importance


class ModelTrainer:
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        label_smoothing = config.get('label_smoothing', 0.0)
        
        # Handle class weights for imbalanced datasets
        class_weights = config.get('class_weights', None)
        if class_weights is not None:
            weight_tensor = torch.FloatTensor(class_weights).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.base_lr = config['learning_rate']
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc="Training", unit="batch") as pbar:
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 检查数据是否有NaN或Inf
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    print("警告: 输入数据包含NaN或Inf")
                    continue
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 检查loss是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print("警告: Loss为NaN或Inf，跳过此batch")
                    continue
                
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # 更新进度条（避免 total=0 时除零）
                n_done = pbar.n + 1
                current_loss = total_loss / n_done
                current_acc = (correct / total) if total > 0 else 0.0
                pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")
        
        # 若所有 batch 因 NaN/Inf 被跳过，total=0，避免除零
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = (correct / total) if total > 0 else 0.0
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # 将数据移到GPU
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        if total == 0:
            return 0.0, 0.0
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        # 显存优化：不一次性加载所有数据到GPU
        # 使用Dataset和DataLoader按需加载数据
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        # 验证集也使用batch processing
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 256),
            shuffle=False
        )
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n=== 开始训练 ===")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"{'='*60}")
        
        with tqdm(range(self.config['epochs']), desc="Epochs", unit="epoch") as epoch_pbar:
            for epoch in epoch_pbar:
                # 学习率 warmup（前 warmup_epochs 轮线性增加）
                if self.warmup_epochs and epoch < self.warmup_epochs:
                    lr_scale = (epoch + 1) / self.warmup_epochs
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.base_lr * lr_scale

                print(f"\nEpoch [{epoch+1}/{self.config['epochs']}]")
                print(f"{'='*40}")

                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)

                if epoch >= self.warmup_epochs:
                    self.scheduler.step(val_loss)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # 保存模型到指定路径
                    model_path = 'models/best_model.pth'
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_path)
                    print(f"模型已保存至: {model_path}")
                else:
                    patience_counter += 1
                
                # 显示epoch结果
                print(f"[Epoch {epoch+1}] 结果:")
                print(f"  训练集: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
                print(f"  验证集: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")
                print(f"  最佳验证准确率: {best_val_acc:.4f}")
                
                # 更新epoch进度条
                epoch_pbar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    train_acc=f"{train_acc:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    val_acc=f"{val_acc:.4f}"
                )
                
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"\n早停触发于 Epoch {epoch+1}")
                    break
        
        # 加载最佳模型（如果存在）
        model_path = 'models/best_model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载最佳模型: {model_path}")
        
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        分批预测，避免一次性将全部数据放入GPU导致显存溢出。
        """
        self.model.eval()
        batch_size = self.config.get('batch_size', 256)
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu())
        return torch.cat(all_preds, dim=0).numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        分批预测概率，同样避免显存溢出。
        """
        self.model.eval()
        batch_size = self.config.get('batch_size', 256)
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_proba = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1)
                all_proba.append(proba.cpu())
        return torch.cat(all_proba, dim=0).numpy()


def create_model(model_type: str, input_dim: int, num_classes: int, config: Dict) -> nn.Module:
    if model_type == 'linear':
        return LinearClassifier(input_dim, num_classes)
    elif model_type == 'mlp':
        return MLPClassifier(
            input_dim, 
            hidden_dims=config.get('hidden_dims', [128, 64]),
            num_classes=num_classes,
            dropout=config.get('dropout', 0.3)
        )
    elif model_type == 'transformer':
        d_model = config.get('d_model', 64)
        nhead = config.get('nhead', 4)
        
        # 添加参数兼容性检查
        if d_model % nhead != 0:
            # 自动调整nhead以确保兼容性
            compatible_nheads = [h for h in [8, 4, 2, 1] if d_model % h == 0]
            if compatible_nheads:
                new_nhead = compatible_nheads[0]
                print(f"警告: 自动调整nhead为 {new_nhead} 以兼容d_model={d_model}")
                nhead = new_nhead
            else:
                raise ValueError(f"无法找到与d_model={d_model}兼容的nhead值")
        
        return TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=config.get('num_layers', 1),
            dim_feedforward=config.get('dim_feedforward', 128),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'fttransformer':
        d_model = config.get('d_model', 64)
        nhead = config.get('nhead', 4)
        
        # 添加参数兼容性检查
        if d_model % nhead != 0:
            # 自动调整nhead以确保兼容性
            compatible_nheads = [h for h in [8, 4, 2, 1] if d_model % h == 0]
            if compatible_nheads:
                new_nhead = compatible_nheads[0]
                print(f"警告: 自动调整nhead为 {new_nhead} 以兼容d_model={d_model}")
                nhead = new_nhead
            else:
                raise ValueError(f"无法找到与d_model={d_model}兼容的nhead值")
        
        return FTTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=config.get('num_layers', 2),
            dim_feedforward=config.get('dim_feedforward', 256),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def get_default_config() -> Dict:
    return {
        'epochs': 100,
        'batch_size': 512,
        'learning_rate': 0.001,
        'dropout': 0.3,
        'early_stopping_patience': 15,
        'hidden_dims': [128, 64],
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'label_smoothing': 0.0,
        'weight_decay': 1e-5,
        'warmup_epochs': 0
    }


def train_classifier(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     model_type: str = 'linear',
                     config: Optional[Dict] = None,
                     output_dir: str = 'results',
                     use_class_weights: bool = True) -> Dict:
    """训练分类器：验证集用于早停与模型选择，测试集仅用于最终评估。"""
    set_seed(42)
    device = get_device()

    if config is None:
        config = get_default_config()
    # FT-Transformer 推荐：warmup + label_smoothing 提高准确率
    if model_type == 'fttransformer':
        config.setdefault('warmup_epochs', 5)
        config.setdefault('label_smoothing', 0.1)
        config.setdefault('weight_decay', 1e-4)

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    # Calculate class weights if requested
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        config['class_weights'] = weights.tolist()
        print(f"计算得到的类别权重: {weights}")

    print(f"\n=== 训练 {model_type.upper()} 分类器 ===")
    print(f"输入维度: {input_dim}")
    print(f"类别数: {num_classes}")
    print(f"训练样本数: {len(X_train)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"测试样本数: {len(X_test)}")

    model = create_model(model_type, input_dim, num_classes, config)
    trainer = ModelTrainer(model, config, device)

    history = trainer.train(X_train, y_train, X_val, y_val)

    y_pred = trainer.predict(X_test)
    y_pred_train = trainer.predict(X_train)

    test_metrics = calculate_metrics(y_test, y_pred)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    
    ensure_dir(output_dir)
    
    print_metrics(train_metrics, "训练集性能")
    print_metrics(test_metrics, "测试集性能")
    
    class_names = [f'Severity {i+1}' for i in range(num_classes)]
    print_classification_report(y_test, y_pred, class_names)
    
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(output_dir, f'{model_type}_confusion_matrix.png')
    )
    
    plot_training_history(
        history,
        save_path=os.path.join(output_dir, f'{model_type}_training_history.png')
    )
    
    model_save_path = os.path.join(output_dir, f'{model_type}_model.pth')
    save_model(model, model_save_path)
    
    results = {
        'model': model,
        'history': history,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'y_pred': y_pred,
        'config': config
    }
    
    return results


if __name__ == '__main__':
    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor('../data/US_Accidents_March23.csv')
    data_dict = preprocessor.preprocess()

    model_types = ['linear', 'mlp', 'transformer']

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"训练 {model_type.upper()} 模型")
        print(f"{'='*60}")

        results = train_classifier(
            data_dict['X_train'],
            data_dict['y_train'],
            data_dict['X_val'],
            data_dict['y_val'],
            data_dict['X_test'],
            data_dict['y_test'],
            model_type=model_type,
            output_dir='results'
        )
