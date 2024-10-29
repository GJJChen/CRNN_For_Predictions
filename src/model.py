# src/model.py

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN模型用于网络流量预测，适应输入形状 [用户数, 时间步长, 网络指标]"""

    def __init__(self, input_size, conv_filters=64, lstm_hidden_size=50):
        super(CRNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # LSTM层
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_hidden_size, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(lstm_hidden_size, input_size)

    def forward(self, x):
        # 假设输入x的形状为 [batch, 用户数, 时间步长, 网络指标]
        batch_size, num_users, seq_len, num_metrics = x.size()
        x = x.view(batch_size * num_users, seq_len, num_metrics)  # 转换为 [batch*用户数, 时间步长, 网络指标]
        x = x.transpose(1, 2)  # 变为 [batch*用户数, 网络指标, 时间步长]

        # 卷积层和池化层
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.transpose(1, 2)  # 变为 [batch*用户数, 时间步长, conv_filters]

        # LSTM层
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 获取LSTM最后一个时间步的输出

        # 全连接层
        x = self.fc(x)
        return x.view(batch_size, num_users, -1)  # 变为 [batch, 用户数, 网络指标]
