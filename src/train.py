# src/train.py

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from data_preprocessing import TrafficDataset, load_data
from model import CRNN


def train_model(data_path, sequence_length=30, batch_size=32, epochs=50, lr=0.001):
    # 设置设备为GPU或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载并准备数据集
    data = load_data(data_path)
    dataset = TrafficDataset(data, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型并转移到设备
    input_size = data.shape[2]  # 网络指标数
    model = CRNN(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练过程
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for sequences, targets in tqdm(dataloader, desc="Training Epochs"):
            sequences, targets = sequences.to(device), targets.to(device)  # 将数据转移到设备
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}')

    # 保存模型
    torch.save(model.state_dict(), '../results/checkpoints/crnn_model.pth')

if __name__ == '__main__':
    train_model(data_path='../data/processed_data/tpdata.npy')
