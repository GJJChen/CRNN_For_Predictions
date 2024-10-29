# src/evaluate.py

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from predict import predict
from data_preprocessing import load_data

def evaluate_model(data_path, sequence_length=30):
    # 加载数据和预测结果
    data = load_data(data_path)[:, sequence_length:, :]  # 忽略序列开始部分
    predictions = predict(data_path, sequence_length=sequence_length)

    # 计算每个网络指标的MSE
    mse_scores = []
    for i in range(data.shape[2]):
        mse = mean_squared_error(data[:, :, i].flatten(), predictions[:, :, i].flatten())
        mse_scores.append(mse)
        print(f"网络指标 {i} 的 MSE: {mse}")

    # 可视化第一个用户的真实值和预测值
    plt.plot(data[0, :, 0], label='真实值')
    plt.plot(predictions[0, :, 0], label='预测值')
    plt.legend()
    plt.show()

# 示例运行
# evaluate_model(data_path='../data/processed_data/network_traffic_data.npy')
