import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
# 1. 全局超参数与硬件配置
NUM_SAMPLES = 10000000  # 2000万条数据
SEQ_LENGTH = 300  # 每条数据 300 个点 (15秒 @ 20Hz)
BATCH_SIZE = 16384  # 大BatchSize充分榨干64GB统一内存
EPOCHS = 20  # 数据量够大，10个 Epoch 即可收敛
LEARNING_RATE = 3e-3
# 自动检测 Mac M系列芯片的MPS加速，如果不支持则退回 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f" 当前使用的加速硬件: {device.type.upper()}")
# 2. 高效数据生成器 (伪标签工厂)
# 2. 高效数据生成器 (伪标签工厂 - 终极抗干扰版)
class VitalSignDataset(Dataset):
    def __init__(self, num_samples, seq_length):
        print(f"正在内存中生成 {num_samples} 条带【体动破坏】的微动数据...")
        start_time = time.time()

        # 时间轴 (15秒, 20Hz)
        t = np.linspace(0, 15, seq_length, dtype=np.float32)

        # 批量生成频率 (遵循论文阈值：呼吸 0.1-0.5Hz, 心跳 0.8-2.0Hz)
        resp_freqs = np.random.uniform(0.1, 0.5, (num_samples, 1)).astype(np.float32)
        heart_freqs = np.random.uniform(0.8, 2.0, (num_samples, 1)).astype(np.float32)

        # 基础波形: 呼吸(大振幅) + 心跳(小振幅)
        clean_signals = np.sin(2 * np.pi * resp_freqs * t) + 0.1 * np.sin(2 * np.pi * heart_freqs * t)

        # 核心创新：全向量化注入非线性体动 (Body Movement)
        body_movement = np.zeros((num_samples, seq_length), dtype=np.float32)
        # 随机抽取约 30% 的数据加入巨大的阶跃漂移（模拟翻身、大幅度抬手）
        mask = np.random.rand(num_samples) < 0.3
        num_masked = np.sum(mask)
        if num_masked > 0:
            # 随机生成阶跃发生的时刻 (跳过开头和结尾的一小段)
            step_idx = np.random.randint(50, 250, num_masked)
            # 随机生成阶跃的巨大振幅 (远大于心肺微动)
            step_heights = np.random.uniform(1.0, 3.0, (num_masked, 1)).astype(np.float32)
            # 利用 NumPy 广播机制极速生成阶跃掩码
            col_indices = np.arange(seq_length)
            step_mask = col_indices >= step_idx[:, None]
            # 将干扰注入到指定样本中
            body_movement[mask] = step_mask * step_heights
        # 加入高斯白噪声
        noise = np.random.normal(0, 0.05, (num_samples, seq_length)).astype(np.float32)
        # 合成最终的“被破坏”的信号
        raw_signals = clean_signals + body_movement + noise
        # 关键预处理：Z-Score 标准化 (必须与 dsp_algo.py 推理时完全对齐！)
        mean = np.mean(raw_signals, axis=1, keepdims=True)
        std = np.std(raw_signals, axis=1, keepdims=True) + 1e-5
        signals = (raw_signals - mean) / std

        # 转换为 PyTorch 张量，并调整形状为 (Batch, Channel, Length) -> (10000000, 1, 300)
        self.x_data = torch.from_numpy(signals).unsqueeze(1)

        # 伪标签转换: Hz -> BPM
        labels = np.hstack((resp_freqs * 60, heart_freqs * 60))
        self.y_data = torch.from_numpy(labels)

        print(f"数据生成及标准化完毕！耗时 {time.time() - start_time:.2f} 秒。")
        print(f"内存占用预估: X={self.x_data.element_size() * self.x_data.nelement() / 1e9:.2f} GB")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# 3. 1D-CNN 模型定义 (带 Batch Norm 加速收敛)

class VitalCNN(nn.Module):
    def __init__(self):
        super(VitalCNN, self).__init__()
        # 第一层卷积：提取基础周期特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 第二层卷积：提取高维抽象特征
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 全连接层回归输出
        # 经过两次池化和步长为2的卷积，300长度->150->75->37
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 37, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        # 输出层：2个节点 (一个是呼吸BPM，一个是心跳BPM)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. 训练主循环

def train_model():
    # 实例化数据集与极速数据加载器
    dataset = VitalSignDataset(NUM_SAMPLES, SEQ_LENGTH)
    # num_workers=0 因为数据全在内存中，单线程读取反而比多进程通信快
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VitalCNN().to(device)

    # 使用 MSE 均方误差作为回归损失函数: $L = \frac{1}{N}\sum(y_i - \hat{y}_i)^2$
    criterion = nn.MSELoss()
    # 使用 AdamW 优化器，比传统 Adam 更快更稳
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # 学习率调度器：OneCycleLR，专为快速收敛设计
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(dataloader), epochs=EPOCHS
    )

    print("\n 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 将数据推送到 Mac 的 GPU (MPS)
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播与反向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # 每 200 个 Batch 打印一次进度
            if (batch_idx + 1) % 200 == 0:
                print(
                    f"   Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(dataloader)
        print(f"🟢 Epoch [{epoch + 1}/{EPOCHS}] 完成 | 平均 Loss: {avg_loss:.4f} | 耗时: {epoch_time:.2f} 秒")

    # 保存训练好的权重
    torch.save(model.state_dict(), "vital_cnn_robust.pth")
    print("\n 训练结束！模型已保存为 'vital_cnn_robust.pth'")


if __name__ == "__main__":
    train_model()
