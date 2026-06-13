import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List
from tqdm import tqdm
import os

from utils import get_device, set_seed


class Generator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 512, 256]):
        super(Generator, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128]):
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class GANTrainer:
    def __init__(self, generator: Generator, discriminator: Discriminator, config: Dict, device: torch.device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
        
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'd_acc_real': [],
            'd_acc_fake': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float, float]:
        self.generator.train()
        self.discriminator.train()
        
        total_d_loss = 0
        total_g_loss = 0
        correct_real = 0
        correct_fake = 0
        total_samples = 0
        
        for batch_X, _ in train_loader:
            batch_size = batch_X.size(0)
            total_samples += batch_size
            
            batch_X = batch_X.to(self.device)
            
            # 创建标签
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # 训练判别器
            self.optimizer_D.zero_grad()
            
            # 判别真实样本
            real_outputs = self.discriminator(batch_X)
            d_loss_real = self.criterion(real_outputs, real_labels)
            
            # 生成假样本
            z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
            fake_samples = self.generator(z)
            
            # 判别假样本
            fake_outputs = self.discriminator(fake_samples.detach())
            d_loss_fake = self.criterion(fake_outputs, fake_labels)
            
            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_D.step()
            
            # 训练生成器
            self.optimizer_G.zero_grad()
            
            fake_outputs = self.discriminator(fake_samples)
            g_loss = self.criterion(fake_outputs, real_labels)
            g_loss.backward()
            self.optimizer_G.step()
            
            # 计算准确率
            correct_real += (real_outputs > 0.5).sum().item()
            correct_fake += (fake_outputs < 0.5).sum().item()
            
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
        
        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        d_acc_real = correct_real / total_samples
        d_acc_fake = correct_fake / total_samples
        
        return avg_d_loss, avg_g_loss, d_acc_real, d_acc_fake
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        print(f"\n=== 开始GAN训练 ===")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Learning Rate: {self.config['learning_rate']}")
        print(f"Latent Dimension: {self.config['latent_dim']}")
        print(f"{'='*60}")
        
        with tqdm(range(self.config['epochs']), desc="GAN Training", unit="epoch") as epoch_pbar:
            for epoch in epoch_pbar:
                d_loss, g_loss, d_acc_real, d_acc_fake = self.train_epoch(train_loader)
                
                self.history['d_loss'].append(d_loss)
                self.history['g_loss'].append(g_loss)
                self.history['d_acc_real'].append(d_acc_real)
                self.history['d_acc_fake'].append(d_acc_fake)
                
                epoch_pbar.set_postfix(
                    d_loss=f"{d_loss:.4f}",
                    g_loss=f"{g_loss:.4f}",
                    d_acc_real=f"{d_acc_real:.4f}",
                    d_acc_fake=f"{d_acc_fake:.4f}"
                )
                
                if (epoch + 1) % 10 == 0:
                    print(f"\nEpoch [{epoch+1}/{self.config['epochs']}]")
                    print(f"判别器损失: {d_loss:.4f}")
                    print(f"生成器损失: {g_loss:.4f}")
                    print(f"判别器真实样本准确率: {d_acc_real:.4f}")
                    print(f"判别器假样本准确率: {d_acc_fake:.4f}")
    
    def generate_samples(self, num_samples: int) -> np.ndarray:
        self.generator.eval()
        
        z = torch.randn(num_samples, self.config['latent_dim']).to(self.device)
        
        with torch.no_grad():
            generated_samples = self.generator(z)
        
        return generated_samples.cpu().numpy()
    
    def save_models(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(self.generator.state_dict(), os.path.join(output_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(output_dir, 'discriminator.pth'))
        
        print(f"模型已保存至: {output_dir}")


def create_gan(input_dim: int, config: Dict, device: torch.device) -> Tuple[Generator, Discriminator]:
    generator = Generator(
        input_dim=config['latent_dim'],
        output_dim=input_dim,
        hidden_dims=config.get('generator_hidden_dims', [256, 512, 256])
    )
    
    discriminator = Discriminator(
        input_dim=input_dim,
        hidden_dims=config.get('discriminator_hidden_dims', [256, 128])
    )
    
    return generator, discriminator

def train_gan(X_train: np.ndarray, y_train: np.ndarray, config: Dict, output_dir: str) -> Dict:
    set_seed(42)
    device = get_device()
    
    input_dim = X_train.shape[1]
    
    print(f"\n=== 初始化GAN模型 ===")
    print(f"输入维度: {input_dim}")
    print(f"潜在空间维度: {config['latent_dim']}")
    
    generator, discriminator = create_gan(input_dim, config, device)
    trainer = GANTrainer(generator, discriminator, config, device)
    
    trainer.train(X_train, y_train)
    trainer.save_models(output_dir)
    
    # 生成样本进行测试
    generated_samples = trainer.generate_samples(10)
    
    results = {
        'history': trainer.history,
        'generated_samples_shape': generated_samples.shape,
        'model_path': output_dir
    }
    
    return results
